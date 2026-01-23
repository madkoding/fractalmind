//! Real GGUF to Fractal Model conversion service.
//!
//! This module implements the actual conversion of GGUF model files
//! into a fractal graph structure stored in SurrealDB.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tracing::{info, error, debug};

use crate::db::connection::DatabaseConnection;
use crate::db::queries::{FractalModelNodeRepository, FractalModelRepository};
use crate::graph::raptor::{Raptor, RaptorNode};
use crate::graph::config::RaptorConfig;
use crate::models::llm::fractal_model::{FractalModel, FractalModelNode, FractalModelStatus, LayerInfo, ModelArchitecture};
use crate::models::llm::gguf_parser::{GGUFParser, GGUFTensorInfo};
use crate::models::{EmbeddingVector, EmbeddingModel};

/// GGUF quantization types and their sizes
#[derive(Debug, Clone, Copy)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 29,
}

impl GGMLType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2_K),
            11 => Some(GGMLType::Q3_K),
            12 => Some(GGMLType::Q4_K),
            13 => Some(GGMLType::Q5_K),
            14 => Some(GGMLType::Q6_K),
            15 => Some(GGMLType::Q8_K),
            29 => Some(GGMLType::BF16),
            _ => None,
        }
    }
}

/// Represents a layer group in the model
#[derive(Debug, Clone)]
pub struct LayerGroup {
    pub layer_start: u32,
    pub layer_end: u32,
    pub layer_type: String,
    pub tensors: Vec<String>,
    pub total_params: u64,
    pub embedding: Vec<f32>,
}

/// Service for converting GGUF models to fractal structure
pub struct ModelConversionService {
    db: Arc<DatabaseConnection>,
    embedding_dim: usize,
}

impl ModelConversionService {
    pub fn new(db: Arc<DatabaseConnection>) -> Self {
        Self { 
            db,
            embedding_dim: 768,
        }
    }

    pub fn with_embedding_dim(db: Arc<DatabaseConnection>, embedding_dim: usize) -> Self {
        Self { db, embedding_dim }
    }

    /// Main conversion function - converts a GGUF model to fractal structure
    pub async fn convert_model(&self, model: &mut FractalModel) -> Result<()> {
        info!("Starting REAL conversion of model: {} ({})", model.name, model.id);
        let model_id = model.id.clone();
        let file_path = model.file_path.clone();

        model.update_status(FractalModelStatus::Converting);
        model.update_conversion_progress(0.0, Some("Initializing".to_string()));
        self.save_model(model).await?;

        let repo = FractalModelRepository::new(&self.db);
        let node_repo = FractalModelNodeRepository::new(&self.db);

        // Phase 1: Parse GGUF (0-10%)
        info!("Phase 1: Parsing GGUF structure for model {}", model_id);
        model.update_conversion_progress(5.0, Some("Parsing GGUF header".to_string()));
        self.save_model(model).await?;

        let (tensors, architecture, data_offset) = match self.parse_gguf_structure(&file_path).await {
            Ok(result) => result,
            Err(e) => {
                error!("Failed to parse GGUF file: {}", e);
                model.update_status(FractalModelStatus::Failed);
                model.update_conversion_progress(0.0, Some(format!("Failed: {}", e)));
                self.save_model(model).await?;
                return Err(e);
            }
        };

        model.set_architecture(architecture.clone());
        model.update_conversion_progress(10.0, Some("Architecture extracted".to_string()));
        self.save_model(model).await?;
        repo.update_architecture(&model_id, &architecture).await?;
        
        info!("Found {} tensors, {} layers, data offset: {}", tensors.len(), architecture.n_layers, data_offset);

        // Phase 2: Group tensors (10-20%)
        info!("Phase 2: Grouping tensors by layer");
        model.update_conversion_progress(15.0, Some("Grouping layers".to_string()));
        self.save_model(model).await?;

        let layer_groups = self.group_tensors_by_layer(&tensors, architecture.n_layers);
        
        model.update_conversion_progress(20.0, Some(format!("{} layer groups created", layer_groups.len())));
        self.save_model(model).await?;
        info!("Created {} layer groups", layer_groups.len());

        // Phase 3: Generate embeddings (20-55%)
        info!("Phase 3: Generating layer embeddings from tensor data");
        model.update_conversion_progress(25.0, Some("Generating embeddings".to_string()));
        self.save_model(model).await?;

        let layer_embeddings = match self.generate_layer_embeddings_blocking(
            &file_path,
            &tensors,
            &layer_groups,
            data_offset,
        ).await {
            Ok(embeddings) => embeddings,
            Err(e) => {
                error!("Failed to generate embeddings: {}", e);
                model.update_status(FractalModelStatus::Failed);
                model.update_conversion_progress(0.0, Some(format!("Failed: {}", e)));
                self.save_model(model).await?;
                return Err(e);
            }
        };

        model.update_conversion_progress(55.0, Some(format!("{} embeddings generated", layer_embeddings.len())));
        self.save_model(model).await?;
        info!("Generated {} layer embeddings", layer_embeddings.len());

        // Phase 4: Build RAPTOR tree (55-75%)
        info!("Phase 4: Building RAPTOR hierarchical tree");
        model.update_conversion_progress(60.0, Some("Building RAPTOR tree".to_string()));
        self.save_model(model).await?;

        let raptor_nodes: Vec<RaptorNode> = layer_embeddings
            .iter()
            .enumerate()
            .map(|(i, group)| {
                // Convert Vec<f32> to EmbeddingVector for RAPTOR
                let embedding_vector = EmbeddingVector::new(
                    group.embedding.clone(),
                    EmbeddingModel::Custom(format!("fractal-model-{}", model.name)),
                );
                RaptorNode::new(
                    format!("layer_group_{}", i),
                    format!(
                        "Layers {}-{} ({}): {} tensors, {} params",
                        group.layer_start,
                        group.layer_end,
                        group.layer_type,
                        group.tensors.len(),
                        format_params(group.total_params)
                    ),
                    embedding_vector,
                )
            })
            .collect();

        let raptor_config = RaptorConfig {
            min_cluster_size: 2,
            max_cluster_size: std::cmp::max(4, layer_embeddings.len() / 3),
            similarity_threshold: 0.55,
            max_depth: 4,
            ..Default::default()
        };

        let raptor = Raptor::new(raptor_config);
        let raptor_tree = raptor.build_tree(raptor_nodes);

        model.update_conversion_progress(75.0, Some(format!(
            "RAPTOR tree: {} nodes, depth {}",
            raptor_tree.nodes.len(),
            raptor_tree.max_depth
        )));
        self.save_model(model).await?;
        
        info!(
            "RAPTOR tree built: {} nodes, {} roots, depth {}",
            raptor_tree.nodes.len(),
            raptor_tree.roots.len(),
            raptor_tree.max_depth
        );

        // Phase 5: Store nodes (75-95%)
        info!("Phase 5: Storing fractal nodes in database");
        model.update_conversion_progress(80.0, Some("Storing nodes".to_string()));
        self.save_model(model).await?;

        let root_node_id = match self.store_fractal_nodes(
            &model_id,
            &raptor_tree,
            &layer_embeddings,
            &node_repo,
        ).await {
            Ok(root_id) => root_id,
            Err(e) => {
                error!("Failed to store nodes: {}", e);
                model.update_status(FractalModelStatus::Failed);
                model.update_conversion_progress(0.0, Some(format!("Failed: {}", e)));
                self.save_model(model).await?;
                return Err(e);
            }
        };

        model.update_conversion_progress(95.0, Some("Nodes stored".to_string()));
        self.save_model(model).await?;

        // Phase 6: Finalize (95-100%)
        info!("Phase 6: Finalizing conversion");
        model.set_root_node(root_node_id.clone());
        repo.set_root_node(&model_id, &root_node_id).await?;

        model.update_status(FractalModelStatus::Ready);
        model.update_conversion_progress(100.0, Some("Complete".to_string()));
        self.save_model(model).await?;
        repo.update_status(&model_id, FractalModelStatus::Ready).await?;

        info!("Model conversion COMPLETED for {} - root node: {}", model.name, root_node_id);
        Ok(())
    }

    async fn parse_gguf_structure(
        &self,
        file_path: &str,
    ) -> Result<(Vec<GGUFTensorInfo>, ModelArchitecture, u64)> {
        if !Path::new(file_path).exists() {
            return Err(anyhow::anyhow!("GGUF file not found: {}", file_path));
        }

        let path = file_path.to_string();
        let result = tokio::task::spawn_blocking(move || {
            let parser = GGUFParser::parse_file(&path)?;
            let architecture = parser.extract_architecture()?;
            let tensors = parser.get_tensors().to_vec();
            let data_offset = parser.get_data_offset();
            Ok::<_, anyhow::Error>((tensors, architecture, data_offset))
        })
        .await
        .context("Failed to spawn blocking task")?
        .context("Failed to parse GGUF file")?;

        Ok(result)
    }
    fn group_tensors_by_layer(
        &self,
        tensors: &[GGUFTensorInfo],
        n_layers: u32,
    ) -> Vec<LayerGroup> {
        let mut groups: Vec<LayerGroup> = Vec::new();
        let mut layer_tensors: HashMap<u32, Vec<&GGUFTensorInfo>> = HashMap::new();
        let mut special_tensors: Vec<&GGUFTensorInfo> = Vec::new();

        for tensor in tensors {
            if let Some(layer_num) = self.extract_layer_number(&tensor.name) {
                layer_tensors.entry(layer_num).or_default().push(tensor);
            } else {
                special_tensors.push(tensor);
            }
        }

        if !special_tensors.is_empty() {
            let total_params: u64 = special_tensors
                .iter()
                .map(|t| t.dimensions.iter().product::<u64>())
                .sum();
            
            groups.push(LayerGroup {
                layer_start: 0,
                layer_end: 0,
                layer_type: "embedding".to_string(),
                tensors: special_tensors.iter().map(|t| t.name.clone()).collect(),
                total_params,
                embedding: vec![0.0; self.embedding_dim],
            });
        }

        let layers_per_group = std::cmp::max(1, n_layers / 6);
        
        for group_idx in 0..((n_layers + layers_per_group - 1) / layers_per_group) {
            let start = group_idx * layers_per_group;
            let end = std::cmp::min(start + layers_per_group - 1, n_layers.saturating_sub(1));
            
            let mut group_tensors: Vec<String> = Vec::new();
            let mut total_params: u64 = 0;
            let mut has_attn = false;
            let mut has_ffn = false;

            for layer in start..=end {
                if let Some(tensors) = layer_tensors.get(&layer) {
                    for tensor in tensors {
                        group_tensors.push(tensor.name.clone());
                        total_params += tensor.dimensions.iter().product::<u64>();
                        
                        if tensor.name.contains("attn") || tensor.name.contains("attention") {
                            has_attn = true;
                        }
                        if tensor.name.contains("ffn") || tensor.name.contains("mlp") {
                            has_ffn = true;
                        }
                    }
                }
            }

            let layer_type = if has_attn && has_ffn {
                "transformer".to_string()
            } else if has_attn {
                "attention".to_string()
            } else if has_ffn {
                "feedforward".to_string()
            } else {
                "mixed".to_string()
            };

            if !group_tensors.is_empty() {
                groups.push(LayerGroup {
                    layer_start: start,
                    layer_end: end,
                    layer_type,
                    tensors: group_tensors,
                    total_params,
                    embedding: vec![0.0; self.embedding_dim],
                });
            }
        }

        groups
    }

    fn extract_layer_number(&self, name: &str) -> Option<u32> {
        let patterns = ["blk.", "layers.", "h.", "block.", "layer."];
        
        for pattern in patterns {
            if let Some(pos) = name.find(pattern) {
                let after_pattern = &name[pos + pattern.len()..];
                if let Some(end) = after_pattern.find('.') {
                    if let Ok(num) = after_pattern[..end].parse::<u32>() {
                        return Some(num);
                    }
                }
            }
        }
        None
    }

    async fn generate_layer_embeddings_blocking(
        &self,
        file_path: &str,
        tensors: &[GGUFTensorInfo],
        layer_groups: &[LayerGroup],
        data_offset: u64,
    ) -> Result<Vec<LayerGroup>> {
        let file_path = file_path.to_string();
        let tensors = tensors.to_vec();
        let layer_groups = layer_groups.to_vec();
        let embedding_dim = self.embedding_dim;

        tokio::task::spawn_blocking(move || {
            Self::generate_layer_embeddings_sync(
                &file_path,
                &tensors,
                &layer_groups,
                embedding_dim,
                data_offset,
            )
        })
        .await
        .context("Spawn blocking failed")?
    }

    fn generate_layer_embeddings_sync(
        file_path: &str,
        tensors: &[GGUFTensorInfo],
        layer_groups: &[LayerGroup],
        embedding_dim: usize,
        data_offset: u64,
    ) -> Result<Vec<LayerGroup>> {
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        info!("Reading tensor data from offset {} (file size: {})", data_offset, mmap.len());
        
        let mut result_groups: Vec<LayerGroup> = Vec::new();
        let tensor_map: HashMap<&str, &GGUFTensorInfo> = tensors
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        for group in layer_groups {
            let embedding = Self::generate_group_embedding(
                &mmap,
                &tensor_map,
                &group.tensors,
                data_offset,
                embedding_dim,
            );

            result_groups.push(LayerGroup {
                layer_start: group.layer_start,
                layer_end: group.layer_end,
                layer_type: group.layer_type.clone(),
                tensors: group.tensors.clone(),
                total_params: group.total_params,
                embedding,
            });
        }

        Ok(result_groups)
    }

    fn generate_group_embedding(
        mmap: &Mmap,
        tensor_map: &HashMap<&str, &GGUFTensorInfo>,
        group_tensor_names: &[String],
        data_offset: u64,
        embedding_dim: usize,
    ) -> Vec<f32> {
        let mut embedding = vec![0.0f32; embedding_dim];
        let mut sample_count = 0;
        let mut tensors_sampled = 0;
        let mut total_samples = 0;

        for tensor_name in group_tensor_names {
            if let Some(tensor) = tensor_map.get(tensor_name.as_str()) {
                let samples = Self::sample_tensor_values(mmap, tensor, data_offset);
                
                if !samples.is_empty() {
                    tensors_sampled += 1;
                    total_samples += samples.len();
                }
                
                for (i, &sample) in samples.iter().enumerate() {
                    let idx = (i * 31 + tensor_name.len()) % embedding_dim;
                    embedding[idx] += sample;
                    sample_count += 1;
                }
            }
        }

        // Log sampling stats for debugging
        if tensors_sampled == 0 {
            tracing::warn!(
                "No samples collected from {} tensors in group! data_offset={}", 
                group_tensor_names.len(), 
                data_offset
            );
        } else {
            tracing::debug!(
                "Sampled {} values from {}/{} tensors",
                total_samples,
                tensors_sampled,
                group_tensor_names.len()
            );
        }

        if sample_count > 0 {
            let scale = 1.0 / (sample_count as f32).sqrt();
            for val in &mut embedding {
                *val *= scale;
            }
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }

    fn sample_tensor_values(
        mmap: &Mmap,
        tensor: &GGUFTensorInfo,
        data_offset: u64,
    ) -> Vec<f32> {
        let mut samples = Vec::new();
        let tensor_offset = (data_offset + tensor.offset) as usize;
        let tensor_size: u64 = tensor.dimensions.iter().product();
        
        // Check if offset is within file bounds
        if tensor_offset >= mmap.len() {
            tracing::warn!(
                "Tensor '{}' offset {} is beyond file size {}",
                tensor.name,
                tensor_offset,
                mmap.len()
            );
            return samples;
        }

        let num_samples = std::cmp::min(256, tensor_size as usize);
        let step = std::cmp::max(1, tensor_size as usize / num_samples);

        match GGMLType::from_u32(tensor.tensor_type) {
            Some(GGMLType::F32) => {
                for i in (0..tensor_size as usize).step_by(step).take(num_samples) {
                    let pos = tensor_offset + i * 4;
                    if pos + 4 <= mmap.len() {
                        let mut cursor = Cursor::new(&mmap[pos..pos + 4]);
                        if let Ok(val) = cursor.read_f32::<LittleEndian>() {
                            if val.is_finite() && val.abs() < 1e6 {
                                samples.push(val);
                            }
                        }
                    }
                }
            }
            Some(GGMLType::F16) | Some(GGMLType::BF16) => {
                for i in (0..tensor_size as usize).step_by(step).take(num_samples) {
                    let pos = tensor_offset + i * 2;
                    if pos + 2 <= mmap.len() {
                        let mut cursor = Cursor::new(&mmap[pos..pos + 2]);
                        if let Ok(val) = cursor.read_u16::<LittleEndian>() {
                            let f32_val = f16::from_bits(val).to_f32();
                            if f32_val.is_finite() {
                                samples.push(f32_val);
                            }
                        }
                    }
                }
            }
            Some(GGMLType::Q4_0) | Some(GGMLType::Q4_1) | Some(GGMLType::Q4_K) => {
                Self::sample_q4_tensor(mmap, tensor_offset, tensor_size, step, &mut samples);
            }
            Some(GGMLType::Q8_0) | Some(GGMLType::Q8_1) | Some(GGMLType::Q8_K) => {
                Self::sample_q8_tensor(mmap, tensor_offset, tensor_size, step, &mut samples);
            }
            _ => {
                for i in 0..std::cmp::min(64, num_samples) {
                    let pos = tensor_offset + i * step;
                    if pos + 4 <= mmap.len() {
                        let mut cursor = Cursor::new(&mmap[pos..pos + 4]);
                        if let Ok(val) = cursor.read_f32::<LittleEndian>() {
                            let clamped = val.clamp(-1e6, 1e6);
                            if clamped.is_finite() {
                                samples.push(clamped);
                            }
                        }
                    }
                }
            }
        }

        samples
    }

    fn sample_q4_tensor(mmap: &Mmap, offset: usize, size: u64, step: usize, samples: &mut Vec<f32>) {
        let block_size = 32;
        let bytes_per_block = 18;
        let num_blocks = (size as usize + block_size - 1) / block_size;

        for block_idx in (0..num_blocks).step_by(std::cmp::max(1, step / block_size)).take(64) {
            let block_offset = offset + block_idx * bytes_per_block;
            if block_offset + bytes_per_block <= mmap.len() {
                let mut cursor = Cursor::new(&mmap[block_offset..block_offset + 2]);
                if let Ok(scale_bits) = cursor.read_u16::<LittleEndian>() {
                    let scale = f16::from_bits(scale_bits).to_f32();
                    
                    for i in 0..4 {
                        let byte_pos = block_offset + 2 + i;
                        if byte_pos < mmap.len() {
                            let byte = mmap[byte_pos];
                            let v0 = ((byte & 0x0F) as i8 - 8) as f32 * scale;
                            let v1 = ((byte >> 4) as i8 - 8) as f32 * scale;
                            if v0.is_finite() { samples.push(v0); }
                            if v1.is_finite() { samples.push(v1); }
                        }
                    }
                }
            }
        }
    }

    fn sample_q8_tensor(mmap: &Mmap, offset: usize, size: u64, step: usize, samples: &mut Vec<f32>) {
        let block_size = 32;
        let bytes_per_block = 34;
        let num_blocks = (size as usize + block_size - 1) / block_size;

        for block_idx in (0..num_blocks).step_by(std::cmp::max(1, step / block_size)).take(64) {
            let block_offset = offset + block_idx * bytes_per_block;
            if block_offset + bytes_per_block <= mmap.len() {
                let mut cursor = Cursor::new(&mmap[block_offset..block_offset + 2]);
                if let Ok(scale_bits) = cursor.read_u16::<LittleEndian>() {
                    let scale = f16::from_bits(scale_bits).to_f32();
                    
                    for i in 0..8 {
                        let pos = block_offset + 2 + i;
                        if pos < mmap.len() {
                            let q = mmap[pos] as i8;
                            let val = q as f32 * scale;
                            if val.is_finite() { samples.push(val); }
                        }
                    }
                }
            }
        }
    }

    async fn store_fractal_nodes(
        &self,
        model_id: &str,
        raptor_tree: &crate::graph::raptor::RaptorTree,
        layer_groups: &[LayerGroup],
        node_repo: &FractalModelNodeRepository<'_>,
    ) -> Result<String> {
        let mut id_map: HashMap<String, String> = HashMap::new();

        // First pass: create leaf nodes
        debug!("Processing {} leaves: {:?}", raptor_tree.leaves.len(), raptor_tree.leaves);
        for leaf_id in &raptor_tree.leaves {
            if let Some(tree_node) = raptor_tree.nodes.get(leaf_id) {
                // For leaf nodes, members contains the original node IDs (layer_group_X)
                // Use the first member to get the original layer group index
                let original_id = tree_node.members.first().cloned().unwrap_or_default();
                let layer_idx = original_id
                    .strip_prefix("layer_group_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                
                debug!("Leaf '{}' (original: '{}') -> layer_idx={}, layer_groups.len={}", 
                       leaf_id, original_id, layer_idx, layer_groups.len());

                let layer_group = layer_groups.get(layer_idx);
                
                if let Some(g) = layer_group {
                    debug!("  Found group: layer_start={}, layer_end={}, layer_type={}", g.layer_start, g.layer_end, g.layer_type);
                } else {
                    debug!("  NO GROUP FOUND for layer_idx={}", layer_idx);
                }
                
                let layer_info = LayerInfo {
                    layer_range: layer_group.map(|g| (g.layer_start, g.layer_end)).unwrap_or((0, 0)),
                    layer_type: layer_group.map(|g| g.layer_type.clone()).unwrap_or_else(|| "unknown".to_string()),
                    summary: tree_node.combined_content.chars().take(500).collect(),
                    metadata: serde_json::json!({
                        "tensor_count": layer_group.map(|g| g.tensors.len()).unwrap_or(0),
                        "total_params": layer_group.map(|g| g.total_params).unwrap_or(0),
                        "internal_similarity": tree_node.internal_similarity,
                    }),
                };

                // Extract Vec<f32> from EmbeddingVector for storage
                let node = FractalModelNode::new(
                    model_id.to_string(),
                    tree_node.centroid.vector.clone(),
                    layer_info,
                    None,
                    0,
                );

                let node_id = node_repo.create(&node).await?;
                id_map.insert(leaf_id.clone(), node_id);
            }
        }

        // Second pass: create parent nodes
        for depth in 1..=raptor_tree.max_depth {
            for (cluster_id, tree_node) in &raptor_tree.nodes {
                if tree_node.depth != depth {
                    continue;
                }

                let children_db_ids: Vec<String> = tree_node
                    .children
                    .iter()
                    .filter_map(|child_id| id_map.get(child_id).cloned())
                    .collect();

                let layer_info = LayerInfo {
                    layer_range: (0, 0),
                    layer_type: format!("cluster_L{}", depth),
                    summary: tree_node
                        .summary
                        .clone()
                        .unwrap_or_else(|| format!("Cluster of {} nodes at level {}", children_db_ids.len(), depth)),
                    metadata: serde_json::json!({
                        "child_count": children_db_ids.len(),
                        "internal_similarity": tree_node.internal_similarity,
                    }),
                };

                // Extract Vec<f32> from EmbeddingVector for storage
                let mut node = FractalModelNode::new(
                    model_id.to_string(),
                    tree_node.centroid.vector.clone(),
                    layer_info,
                    None,
                    depth as u32,
                );

                for child_id in &children_db_ids {
                    node.add_child(child_id.clone());
                }

                let node_id = node_repo.create(&node).await?;
                id_map.insert(cluster_id.clone(), node_id.clone());

                for child_cluster_id in &tree_node.children {
                    if let Some(child_db_id) = id_map.get(child_cluster_id) {
                        node_repo.update_parent(child_db_id, &node_id).await?;
                    }
                }
            }
        }

        let root_id = raptor_tree
            .roots
            .first()
            .and_then(|root_cluster| id_map.get(root_cluster))
            .cloned()
            .unwrap_or_else(|| {
                id_map.values().next().cloned().unwrap_or_else(|| "unknown".to_string())
            });

        Ok(root_id)
    }

    async fn save_model(&self, model: &FractalModel) -> Result<()> {
        let id_part = model.id.strip_prefix("fractal_models:").unwrap_or(&model.id);
        
        let query = r#"
            UPDATE type::thing("fractal_models", $id) SET
                name = $name,
                architecture = $architecture,
                root_node_id = $root_node_id,
                status = $status,
                file_path = $file_path,
                file_size = $file_size,
                conversion_progress = $conversion_progress,
                conversion_phase = $conversion_phase,
                updated_at = time::now(),
                metadata = $metadata
        "#;

        self.db
            .query(query)
            .bind(("id", id_part))
            .bind(("name", model.name.clone()))
            .bind(("architecture", serde_json::to_value(&model.architecture)?))
            .bind(("root_node_id", model.root_node_id.clone()))
            .bind(("status", format!("{:?}", model.status).to_lowercase()))
            .bind(("file_path", model.file_path.clone()))
            .bind(("file_size", model.file_size))
            .bind(("conversion_progress", model.conversion_progress))
            .bind(("conversion_phase", model.conversion_phase.clone()))
            .bind(("metadata", model.metadata.clone()))
            .await
            .context("Failed to save model to database")?;

        Ok(())
    }

    pub async fn create_model(&self, model: &FractalModel) -> Result<()> {
        let id_part = model.id.strip_prefix("fractal_models:").unwrap_or(&model.id);
        
        let query = r#"
            CREATE type::thing("fractal_models", $id) SET
                name = $name,
                architecture = $architecture,
                root_node_id = $root_node_id,
                status = $status,
                file_path = $file_path,
                file_size = $file_size,
                conversion_progress = $conversion_progress,
                conversion_phase = $conversion_phase,
                created_at = time::now(),
                updated_at = time::now(),
                metadata = $metadata
        "#;

        self.db
            .query(query)
            .bind(("id", id_part))
            .bind(("name", model.name.clone()))
            .bind(("architecture", serde_json::to_value(&model.architecture)?))
            .bind(("root_node_id", model.root_node_id.clone()))
            .bind(("status", format!("{:?}", model.status).to_lowercase()))
            .bind(("file_path", model.file_path.clone()))
            .bind(("file_size", model.file_size))
            .bind(("conversion_progress", model.conversion_progress))
            .bind(("conversion_phase", model.conversion_phase.clone()))
            .bind(("metadata", model.metadata.clone()))
            .await
            .context("Failed to create model in database")?;

        Ok(())
    }

    pub async fn list_models(&self) -> Result<Vec<FractalModel>> {
        let query = "SELECT * FROM fractal_models ORDER BY created_at DESC";
        
        let mut response = self.db
            .query(query)
            .await
            .context("Failed to list models")?;

        let models: Vec<FractalModel> = response.take(0)?;
        Ok(models)
    }

    pub async fn get_model(&self, model_id: &str) -> Result<Option<FractalModel>> {
        let id_part = model_id.strip_prefix("fractal_models:").unwrap_or(model_id);
        let query = "SELECT * FROM type::thing(\"fractal_models\", $id)";
        
        let mut response = self.db
            .query(query)
            .bind(("id", id_part))
            .await
            .context("Failed to get model")?;

        let models: Vec<FractalModel> = response.take(0)?;
        Ok(models.into_iter().next())
    }

    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        if let Some(model) = self.get_model(model_id).await? {
            if Path::new(&model.file_path).exists() {
                fs::remove_file(&model.file_path)
                    .await
                    .context("Failed to delete GGUF file")?;
            }
        }

        let delete_nodes_query = "DELETE fractal_model_nodes WHERE model_id = $model_id";
        self.db
            .query(delete_nodes_query)
            .bind(("model_id", model_id))
            .await?;

        let id_part = model_id.strip_prefix("fractal_models:").unwrap_or(model_id);
        let delete_model_query = "DELETE type::thing(\"fractal_models\", $id)";
        self.db
            .query(delete_model_query)
            .bind(("id", id_part))
            .await
            .context("Failed to delete model from database")?;

        Ok(())
    }
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        format!("{}", params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(1500000000), "1.5B");
        assert_eq!(format_params(7000000), "7.0M");
        assert_eq!(format_params(1500), "1.5K");
        assert_eq!(format_params(500), "500");
    }

    #[test]
    fn test_ggml_type_conversion() {
        assert!(GGMLType::from_u32(0).is_some());
        assert!(GGMLType::from_u32(12).is_some());
        assert!(GGMLType::from_u32(255).is_none());
    }
}
