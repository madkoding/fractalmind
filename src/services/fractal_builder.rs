//! Fractal structure building service.
//!
//! This module takes leaf nodes and builds the hierarchical fractal structure
//! using RAPTOR clustering algorithm.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use surrealdb::sql::Thing;
use tracing::{info, warn, debug};

use crate::db::connection::DatabaseConnection;
use crate::db::queries::{NodeRepository, EdgeRepository};
use crate::graph::{Raptor, RaptorConfig, RaptorNode, RaptorTree, RaptorTreeNode};
use crate::models::{FractalNode, FractalEdge, NodeMetadata, SourceType};
use crate::models::llm::ModelBrain;

/// Configuration for fractal building
#[derive(Debug, Clone)]
pub struct FractalBuilderConfig {
    /// RAPTOR configuration
    pub raptor_config: RaptorConfig,
    /// Whether to generate summaries using LLM
    pub generate_summaries: bool,
    /// Maximum summary length
    pub max_summary_length: usize,
    /// Minimum nodes required to build fractal structure
    pub min_nodes_for_fractal: usize,
}

impl Default for FractalBuilderConfig {
    fn default() -> Self {
        Self {
            raptor_config: RaptorConfig::default(),
            generate_summaries: true,
            max_summary_length: 500,
            min_nodes_for_fractal: 5,
        }
    }
}

impl FractalBuilderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_raptor_config(mut self, config: RaptorConfig) -> Self {
        self.raptor_config = config;
        self
    }

    pub fn with_summaries(mut self, enabled: bool) -> Self {
        self.generate_summaries = enabled;
        self
    }

    pub fn with_min_nodes(mut self, min: usize) -> Self {
        self.min_nodes_for_fractal = min;
        self
    }
}

/// Result of fractal building operation
#[derive(Debug)]
pub struct FractalBuildResult {
    /// Number of parent nodes created
    pub parent_nodes_created: usize,
    /// Number of edges created
    pub edges_created: usize,
    /// Maximum depth of the fractal tree
    pub max_depth: usize,
    /// Total time in milliseconds
    pub build_time_ms: u64,
    /// Root node IDs
    pub root_node_ids: Vec<String>,
}

/// Service for building fractal hierarchical structures
pub struct FractalBuilder<'a> {
    db: &'a DatabaseConnection,
    config: FractalBuilderConfig,
}

impl<'a> FractalBuilder<'a> {
    /// Creates a new fractal builder
    pub fn new(db: &'a DatabaseConnection, config: FractalBuilderConfig) -> Self {
        Self { db, config }
    }

    /// Creates a fractal builder with default configuration
    pub fn with_defaults(db: &'a DatabaseConnection) -> Self {
        Self::new(db, FractalBuilderConfig::default())
    }

    /// Builds fractal structure for nodes in a namespace
    pub async fn build_for_namespace(
        &self,
        namespace: &str,
        brain: Option<&ModelBrain>,
    ) -> Result<FractalBuildResult> {
        let start = Instant::now();
        let node_repo = NodeRepository::new(self.db);
        let edge_repo = EdgeRepository::new(self.db);

        // 1. Fetch all leaf nodes in the namespace
        let leaf_nodes = self.fetch_leaf_nodes(namespace).await?;
        
        if leaf_nodes.len() < self.config.min_nodes_for_fractal {
            info!(
                "Skipping fractal build for namespace '{}': only {} nodes (min: {})",
                namespace, leaf_nodes.len(), self.config.min_nodes_for_fractal
            );
            return Ok(FractalBuildResult {
                parent_nodes_created: 0,
                edges_created: 0,
                max_depth: 0,
                build_time_ms: start.elapsed().as_millis() as u64,
                root_node_ids: vec![],
            });
        }

        info!("Building fractal structure for {} leaf nodes in namespace '{}'", 
              leaf_nodes.len(), namespace);

        // 2. Convert to RAPTOR nodes
        let raptor_nodes: Vec<RaptorNode> = leaf_nodes
            .iter()
            .filter_map(|node| {
                node.id.as_ref().map(|id| {
                    RaptorNode::new(
                        id.to_string(),
                        node.content.clone(),
                        node.embedding.clone(),
                    )
                })
            })
            .collect();

        // 3. Build RAPTOR tree
        let raptor = Raptor::new(self.config.raptor_config.clone());
        let raptor_tree = raptor.build_tree(raptor_nodes);

        info!(
            "RAPTOR tree built: {} clusters, max_depth={}, time={}ms",
            raptor_tree.stats.total_clusters,
            raptor_tree.max_depth,
            raptor_tree.build_time_ms
        );

        // 4. Create parent nodes and edges from RAPTOR tree
        let (parent_nodes_created, edges_created, root_ids) = self
            .persist_raptor_tree(&raptor_tree, namespace, brain, &node_repo, &edge_repo)
            .await?;

        let result = FractalBuildResult {
            parent_nodes_created,
            edges_created,
            max_depth: raptor_tree.max_depth,
            build_time_ms: start.elapsed().as_millis() as u64,
            root_node_ids: root_ids,
        };

        info!(
            "Fractal structure built: {} parent nodes, {} edges, max_depth={}, time={}ms",
            result.parent_nodes_created,
            result.edges_created,
            result.max_depth,
            result.build_time_ms
        );

        Ok(result)
    }

    /// Fetches all leaf nodes for a namespace
    async fn fetch_leaf_nodes(&self, namespace: &str) -> Result<Vec<FractalNode>> {
        let query = r#"
            SELECT * FROM nodes 
            WHERE namespace = $namespace 
            AND node_type = 'leaf'
            AND depth_level = 0
        "#;

        let mut result = self.db
            .query(query)
            .bind(("namespace", namespace))
            .await
            .context("Failed to fetch leaf nodes")?;

        let nodes: Vec<FractalNode> = result.take(0)?;
        Ok(nodes)
    }

    /// Persists the RAPTOR tree as FractalNodes and FractalEdges
    async fn persist_raptor_tree(
        &self,
        tree: &RaptorTree,
        namespace: &str,
        brain: Option<&ModelBrain>,
        node_repo: &NodeRepository<'_>,
        edge_repo: &EdgeRepository<'_>,
    ) -> Result<(usize, usize, Vec<String>)> {
        let mut parent_count = 0;
        let mut edge_count = 0;
        let mut cluster_to_node_id: HashMap<String, Thing> = HashMap::new();
        let mut root_ids = Vec::new();

        // Map leaf cluster IDs to actual node IDs
        for leaf_id in &tree.leaves {
            if let Some(tree_node) = tree.nodes.get(leaf_id) {
                // For leaf clusters, members are the actual node IDs
                if let Some(member_id) = tree_node.members.first() {
                    // Parse the member_id as a Thing
                    if let Ok(thing) = self.parse_thing(member_id) {
                        cluster_to_node_id.insert(leaf_id.clone(), thing);
                    }
                }
            }
        }

        // Process nodes by depth level (bottom-up)
        for depth in 1..=tree.max_depth {
            let nodes_at_depth: Vec<(&String, &RaptorTreeNode)> = tree
                .nodes
                .iter()
                .filter(|(_, n)| n.depth == depth)
                .collect();

            debug!("Processing {} nodes at depth {}", nodes_at_depth.len(), depth);

            for (cluster_id, tree_node) in nodes_at_depth {
                // Generate summary if enabled and brain is available
                let summary = if self.config.generate_summaries {
                    self.generate_summary(tree_node, brain).await
                } else {
                    None
                };

                // Create parent node
                let parent_node = self.create_parent_node(
                    tree_node,
                    namespace,
                    depth as u32,
                    summary,
                )?;

                let parent_id = node_repo.create(&parent_node).await?;
                parent_count += 1;

                cluster_to_node_id.insert(cluster_id.clone(), parent_id.clone());

                // Create edges from parent to children
                for child_cluster_id in &tree_node.children {
                    if let Some(child_node_id) = cluster_to_node_id.get(child_cluster_id) {
                        // Calculate similarity between parent and child
                        let similarity = self.calculate_parent_child_similarity(
                            tree_node,
                            child_cluster_id,
                            tree,
                        );

                        let edge = FractalEdge::new_parent_child(
                            parent_id.clone(),
                            child_node_id.clone(),
                            similarity,
                        );

                        edge_repo.create(&edge).await?;
                        edge_count += 1;
                    }
                }

                // Track root nodes
                if tree.roots.contains(cluster_id) {
                    root_ids.push(parent_id.to_string());
                }
            }
        }

        // Create semantic edges between siblings (nodes with same parent)
        let sibling_edges = self.create_sibling_edges(tree, &cluster_to_node_id, edge_repo).await?;
        edge_count += sibling_edges;

        Ok((parent_count, edge_count, root_ids))
    }

    /// Parses a string as a SurrealDB Thing
    fn parse_thing(&self, id: &str) -> Result<Thing> {
        // Handle both "table:id" format and raw IDs
        if id.contains(':') {
            let parts: Vec<&str> = id.split(':').collect();
            if parts.len() == 2 {
                Ok(Thing::from((parts[0].to_string(), parts[1].to_string())))
            } else {
                Ok(Thing::from(("nodes".to_string(), id.to_string())))
            }
        } else {
            Ok(Thing::from(("nodes".to_string(), id.to_string())))
        }
    }

    /// Creates a parent FractalNode from a RAPTOR tree node
    fn create_parent_node(
        &self,
        tree_node: &RaptorTreeNode,
        namespace: &str,
        depth: u32,
        summary: Option<String>,
    ) -> Result<FractalNode> {
        let content = if let Some(ref sum) = summary {
            sum.clone()
        } else {
            // Truncate combined content if too long
            let max_len = self.config.max_summary_length * 2;
            if tree_node.combined_content.len() > max_len {
                format!("{}...", &tree_node.combined_content[..max_len])
            } else {
                tree_node.combined_content.clone()
            }
        };

        let mut metadata = NodeMetadata::default();
        metadata.source = "fractal_builder".to_string();
        metadata.source_type = SourceType::Synthetic;
        metadata.tags = vec![
            format!("depth:{}", depth),
            format!("children:{}", tree_node.children.len()),
            format!("cluster:{}", tree_node.cluster_id),
        ];

        Ok(FractalNode::new_parent(
            summary.unwrap_or_else(|| "Cluster summary".to_string()),
            content,
            tree_node.centroid.clone(),
            depth,
            namespace.to_string(),
            None,
            metadata,
        ))
    }

    /// Generates a summary for a tree node using LLM
    async fn generate_summary(
        &self,
        tree_node: &RaptorTreeNode,
        brain: Option<&ModelBrain>,
    ) -> Option<String> {
        let brain = brain?;

        // Prepare content for summarization
        let content_preview = if tree_node.combined_content.len() > 2000 {
            format!("{}...", &tree_node.combined_content[..2000])
        } else {
            tree_node.combined_content.clone()
        };

        let prompt = format!(
            "Genera un resumen conciso (máximo {} caracteres) del siguiente contenido. \
            El resumen debe capturar los conceptos principales y ser útil para búsqueda semántica:\n\n{}",
            self.config.max_summary_length,
            content_preview
        );

        match brain.simple_chat(&prompt).await {
            Ok(response) => {
                let summary = response.content.trim().to_string();
                if summary.len() <= self.config.max_summary_length {
                    Some(summary)
                } else {
                    Some(summary[..self.config.max_summary_length].to_string())
                }
            }
            Err(e) => {
                warn!("Failed to generate summary: {}", e);
                None
            }
        }
    }

    /// Calculates similarity between parent and child nodes
    fn calculate_parent_child_similarity(
        &self,
        parent: &RaptorTreeNode,
        child_cluster_id: &str,
        tree: &RaptorTree,
    ) -> f32 {
        if let Some(child) = tree.nodes.get(child_cluster_id) {
            crate::graph::similarity::cosine_similarity(
                &parent.centroid,
                &child.centroid,
            )
        } else {
            0.8 // Default high similarity for parent-child
        }
    }

    /// Creates semantic edges between sibling nodes
    async fn create_sibling_edges(
        &self,
        tree: &RaptorTree,
        cluster_to_node_id: &HashMap<String, Thing>,
        edge_repo: &EdgeRepository<'_>,
    ) -> Result<usize> {
        let mut edge_count = 0;
        let similarity_threshold = self.config.raptor_config.similarity_threshold;

        // For each parent node, create edges between its children
        for tree_node in tree.nodes.values() {
            if tree_node.children.len() < 2 {
                continue;
            }

            // Get child centroids
            let children: Vec<(&String, &RaptorTreeNode)> = tree_node
                .children
                .iter()
                .filter_map(|id| tree.nodes.get(id).map(|n| (id, n)))
                .collect();

            // Create edges between highly similar siblings
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    let (id_a, child_a) = children[i];
                    let (id_b, child_b) = children[j];

                    let similarity = crate::graph::similarity::cosine_similarity(
                        &child_a.centroid,
                        &child_b.centroid,
                    );

                    if similarity >= similarity_threshold {
                        if let (Some(node_a), Some(node_b)) = 
                            (cluster_to_node_id.get(id_a), cluster_to_node_id.get(id_b)) 
                        {
                            let edge = FractalEdge::new_semantic(
                                node_a.clone(),
                                node_b.clone(),
                                similarity,
                            );
                            edge_repo.create(&edge).await?;
                            edge_count += 1;
                        }
                    }
                }
            }
        }

        Ok(edge_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = FractalBuilderConfig::default();
        assert!(config.generate_summaries);
        assert_eq!(config.min_nodes_for_fractal, 5);
    }

    #[test]
    fn test_config_builder() {
        let config = FractalBuilderConfig::new()
            .with_summaries(false)
            .with_min_nodes(10);
        
        assert!(!config.generate_summaries);
        assert_eq!(config.min_nodes_for_fractal, 10);
    }
}
