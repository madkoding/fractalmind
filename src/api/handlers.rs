//! API request handlers.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use axum::{extract::{State, Multipart}, Json};
use serde::{Deserialize};
use serde_json;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::services::ingestion::extractors::{ExtractorFactory, ContentExtractor};
use crate::services::ingestion::config::{FileType, IngestionConfig};
use crate::services::ingestion::chunker::TextChunker;
use crate::services::FractalBuilder;
use crate::db::queries::{NodeRepository, EdgeRepository};

use crate::cache::{EmbeddingCache, NodeCache};
use crate::db::connection::DatabaseConnection;
use crate::models::llm::ModelBrain;
use crate::models::{EmbeddingVector, FractalNode, NodeMetadata};

use super::error::{ApiError, ApiResult};
use super::progress::ProgressTracker;
use super::types::*;
use crate::services::UploadSessionManager;

/// Application state shared across handlers
pub struct AppState {
    /// Database connection
    pub db: DatabaseConnection,

    /// LLM model brain
    pub brain: ModelBrain,

    /// Node cache
    pub node_cache: NodeCache,

    /// Embedding cache
    pub embedding_cache: EmbeddingCache,
    
    /// Progress tracker for long-running operations
    pub progress_tracker: ProgressTracker,
    
    /// Upload session manager for chunked model uploads
    pub upload_manager: Arc<UploadSessionManager>,
}

/// Thread-safe shared state
pub type SharedState = Arc<RwLock<AppState>>;

// ============================================================================
// Health Check Handler
// ============================================================================

/// Health check endpoint
pub async fn health_check(State(state): State<SharedState>) -> Json<HealthResponse> {
    let state = state.read().await;

    // Check database connection
    let db_healthy = crate::db::connection::check_connection(&state.db)
        .await
        .unwrap_or(false);

    // Check LLM availability
    let llm_healthy = !state.brain.get_models_info().embedding_model.is_empty();

    // Cache is always available (in-memory)
    let cache_healthy = true;

    let status = if db_healthy && llm_healthy {
        "healthy"
    } else if db_healthy || llm_healthy {
        "degraded"
    } else {
        "unhealthy"
    };

    Json(HealthResponse {
        status: status.to_string(),
        service: "fractal-mind".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        components: HealthComponents {
            database: db_healthy,
            llm: llm_healthy,
            cache: cache_healthy,
        },
    })
}

// ============================================================================
// Ingest Handler
// ============================================================================

/// Ingest content into the knowledge graph
pub async fn ingest(
    State(state): State<SharedState>,
    Json(request): Json<IngestRequest>,
) -> ApiResult<Json<IngestResponse>> {
    let start = Instant::now();

    // Validate request
    if request.content.trim().is_empty() {
        return Err(ApiError::ValidationError("Content cannot be empty".to_string()));
    }

    let state = state.read().await;
    let namespace = request.namespace.unwrap_or_else(|| "global_knowledge".to_string());

    debug!("Ingesting content into namespace: {}", namespace);

    // Check embedding cache first
    if let Some(cached) = state.embedding_cache.get(&request.content) {
        debug!("Using cached embedding");
        let node_id = Uuid::new_v4().to_string();

        return Ok(Json(IngestResponse {
            success: true,
            node_id: Some(node_id),
            embedding_dimension: Some(cached.dimension),
            latency_ms: start.elapsed().as_millis() as u64,
            message: "Content ingested (embedding from cache)".to_string(),
        }));
    }

    // Generate embedding
    let embedding_response = state
        .brain
        .embed(&request.content)
        .await
        .map_err(|e| ApiError::EmbeddingError(e.to_string()))?;

    info!(
        "Generated embedding: {}D, latency: {}ms",
        embedding_response.dimension, embedding_response.latency_ms
    );

    // Cache the embedding
    let embedding_vector = EmbeddingVector::new(
        embedding_response.embedding.clone(),
        crate::models::EmbeddingModel::NomicEmbedTextV15,
    );
    state.embedding_cache.put(&request.content, embedding_vector.clone());

    // Create metadata
    let mut metadata = NodeMetadata::default();
    if let Some(source) = &request.source {
        metadata.source = source.clone();
    }
    if let Some(tags) = &request.tags {
        metadata.tags = tags.clone();
    }
    if let Some(lang) = &request.language {
        metadata.language = lang.clone();
    }

    let namespace_clone = namespace.clone();

    // Create fractal node
    let node = FractalNode::new_leaf(
        request.content.clone(),
        embedding_vector,
        namespace,
        request.source.clone(),
        metadata,
    );

    // Save node to database
    let node_repo = NodeRepository::new(&state.db);
    let node_id = node_repo
        .create(&node)
        .await
        .map_err(|e| ApiError::DatabaseError(format!("Failed to create node: {}", e)))?
        .to_string();

    // Auto-build fractal structure
    let config = crate::services::FractalBuilderConfig::new()
        .with_summaries(false)
        .with_min_nodes(3);
    let fractal_builder = FractalBuilder::new(&state.db, config);
    let fractal_msg = match fractal_builder.build_for_namespace(&namespace_clone, Some(&state.brain)).await {
        Ok(result) if result.parent_nodes_created > 0 => {
            info!(
                "Fractal structure updated: {} parent nodes, {} edges",
                result.parent_nodes_created, result.edges_created
            );
            format!(" + fractal updated ({} parents)", result.parent_nodes_created)
        }
        Ok(_) => String::new(),
        Err(e) => {
            warn!("Failed to build fractal structure: {}", e);
            String::new()
        }
    };

    Ok(Json(IngestResponse {
        success: true,
        node_id: Some(node_id),
        embedding_dimension: Some(embedding_response.dimension),
        latency_ms: start.elapsed().as_millis() as u64,
        message: format!("Content ingested successfully{}", fractal_msg),
    }))
}

/// Ingest a file (multipart form: file, namespace, tags (json), language)
pub async fn ingest_file(
    State(state): State<SharedState>,
    mut multipart: Multipart,
) -> ApiResult<Json<IngestResponse>> {
    let start = Instant::now();

    // Read config for ingestion limits and features
    let config = IngestionConfig::from_env();

    // Parse multipart fields
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut namespace: Option<String> = None;
    let mut tags: Option<Vec<String>> = None;
    let mut language: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::BadRequest(format!("Multipart parse error: {}", e)))?
    {
        match field.name() {
            Some("file") => {
                filename = field.file_name().map(|s| s.to_string());
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read file bytes: {}", e)))?
                    .to_vec();
                if data.is_empty() {
                    return Err(ApiError::ValidationError("Uploaded file is empty".to_string()));
                }
                file_bytes = Some(data);
            }
            Some("namespace") => {
                let txt = field
                    .text()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read namespace: {}", e)))?;
                if !txt.trim().is_empty() {
                    namespace = Some(txt);
                }
            }
            Some("tags") => {
                let txt = field
                    .text()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read tags: {}", e)))?;
                if !txt.trim().is_empty() {
                    // Try parse JSON array first, otherwise treat as single tag
                    match serde_json::from_str::<Vec<String>>(&txt) {
                        Ok(v) => tags = Some(v),
                        Err(_) => tags = Some(vec![txt]),
                    }
                }
            }
            Some("language") => {
                let txt = field
                    .text()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read language: {}", e)))?;
                if !txt.trim().is_empty() {
                    language = Some(txt);
                }
            }
            _ => {
                // Ignore unknown fields
            }
        }
    }

    let file_bytes = file_bytes
        .ok_or_else(|| ApiError::ValidationError("Missing 'file' field in multipart".to_string()))?;

    // Size check
    let file_size = file_bytes.len();
    if !config.is_size_allowed(file_size) {
        return Err(ApiError::ValidationError(format!(
            "File size {} exceeds maximum allowed {} bytes",
            file_size, config.max_file_size
        )));
    }

    // Determine file type
    let file_type = if let Some(fname) = &filename {
        std::path::Path::new(fname)
            .extension()
            .and_then(|s| s.to_str())
            .map(FileType::from_extension)
            .unwrap_or(FileType::Unknown)
    } else {
        FileType::Unknown
    };

    if !file_type.is_supported() {
        return Err(ApiError::ValidationError("Unsupported or unknown file type".to_string()));
    }

    // Choose extractor
    let extractor: Box<dyn ContentExtractor> = match file_type {
        FileType::Text => ExtractorFactory::text(),
        FileType::Pdf => {
            if !config.enable_pdf {
                return Err(ApiError::ValidationError("PDF ingestion is disabled".to_string()));
            }
            ExtractorFactory::create(FileType::Pdf)
                .ok_or_else(|| ApiError::ValidationError("PDF extractor not available".to_string()))?
        }
        FileType::Image => {
            if !config.enable_ocr {
                return Err(ApiError::ValidationError("Image OCR ingestion is disabled".to_string()));
            }
            // Image extractor is feature-gated
            #[cfg(feature = "ocr")]
            {
                ExtractorFactory::image(language.as_deref().unwrap_or(&config.ocr_language))
            }
            #[cfg(not(feature = "ocr"))]
            {
                return Err(ApiError::ValidationError(
                    "Image OCR extractor not compiled in this build".to_string(),
                ));
            }
        }
        _ => return Err(ApiError::ValidationError("Unsupported file type".to_string())),
    };

    // Run extraction
    let extraction = extractor
        .extract(&file_bytes)
        .await
        .map_err(|e| ApiError::InternalError(format!("Extraction failed: {}", e)))?;

    if !extraction.is_successful() {
        return Err(ApiError::ValidationError("Could not extract text from file".to_string()));
    }

    // Use provided namespace or default
    let namespace = namespace.unwrap_or_else(|| "global_knowledge".to_string());

    // Chunk the extracted text to avoid exceeding embedding model context limits
    let chunker = TextChunker::from_config(&config);
    let chunking_result = chunker.chunk(&extraction.text);
    
    info!(
        "Chunked document into {} chunks (original: {} chars, avg chunk: {} chars)",
        chunking_result.count(),
        chunking_result.original_length,
        chunking_result.avg_chunk_size
    );

    // Generate embedding (allow tests to disable real model calls)
    let state = state.read().await;

    let disable_embedding = std::env::var("TEST_DISABLE_EMBEDDING")
        .map(|v| v == "true")
        .unwrap_or(false);

    let skip_db = std::env::var("TEST_SKIP_DB_WRITES").map(|v| v == "true").unwrap_or(false);
    let node_repo = NodeRepository::new(&state.db);

    // Process each chunk
    let mut created_node_ids: Vec<String> = Vec::new();
    let mut embedding_dimension: Option<usize> = None;
    let source_filename = filename.clone().unwrap_or_else(|| "uploaded_file".to_string());

    for chunk in chunking_result.chunks {
        // Generate embedding for this chunk
        let embedding_response = if disable_embedding {
            let dim = std::env::var("EMBEDDING_DIMENSION")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(768);
            crate::models::llm::traits_llm::EmbeddingResponse {
                embedding: vec![0.0f32; dim],
                dimension: dim,
                model: "test-mock".to_string(),
                latency_ms: 0,
            }
        } else {
            state
                .brain
                .embed(&chunk.content)
                .await
                .map_err(|e| ApiError::EmbeddingError(format!(
                    "Embedding failed for chunk {}/{}: {}",
                    chunk.index + 1,
                    chunk.total,
                    e
                )))?
        };

        embedding_dimension = Some(embedding_response.dimension);

        // Cache embedding
        let embedding_vector = EmbeddingVector::new(
            embedding_response.embedding.clone(),
            crate::models::EmbeddingModel::NomicEmbedTextV15,
        );
        state.embedding_cache.put(&chunk.content, embedding_vector.clone());

        // Prepare metadata for this chunk
        let mut metadata = NodeMetadata::default();
        metadata.source = source_filename.clone();
        metadata.source_type = match file_type {
            FileType::Pdf => crate::models::SourceType::Pdf,
            FileType::Image => crate::models::SourceType::Image,
            FileType::Text => crate::models::SourceType::Text,
            _ => crate::models::SourceType::Text,
        };
        metadata.language = extraction
            .metadata
            .language
            .clone()
            .or(language.clone())
            .unwrap_or_else(|| "en".to_string());
        if let Some(t) = &tags {
            metadata.tags = t.clone();
        }
        // Add chunk info to metadata tags
        metadata.tags.push(format!("chunk:{}/{}", chunk.index + 1, chunk.total));

        // Create node for this chunk
        let node = FractalNode::new_leaf(
            chunk.content.clone(),
            embedding_vector,
            namespace.clone(),
            Some(source_filename.clone()),
            metadata,
        );

        // Persist
        let node_id = if skip_db {
            Uuid::new_v4().to_string()
        } else {
            let created = node_repo
                .create(&node)
                .await
                .map_err(|e| ApiError::DatabaseError(format!(
                    "Failed to create node for chunk {}/{}: {}",
                    chunk.index + 1,
                    chunk.total,
                    e
                )))?;
            created.to_string()
        };

        created_node_ids.push(node_id);
    }

    let total_chunks = created_node_ids.len();
    let first_node_id = created_node_ids.first().cloned();

    info!(
        "Ingested file '{}' as {} chunks in namespace '{}'",
        source_filename, total_chunks, namespace
    );

    // Auto-build fractal structure after ingestion
    let fractal_result = if !skip_db && total_chunks > 0 {
        info!("Auto-building fractal structure for namespace '{}'", namespace);
        let config = crate::services::FractalBuilderConfig::new()
            .with_summaries(false)
            .with_min_nodes(3);
        let fractal_builder = FractalBuilder::new(&state.db, config);
        match fractal_builder.build_for_namespace(&namespace, Some(&state.brain)).await {
            Ok(result) => {
                info!(
                    "Fractal structure built: {} parent nodes, {} edges",
                    result.parent_nodes_created, result.edges_created
                );
                Some(result)
            }
            Err(e) => {
                warn!("Failed to build fractal structure: {}", e);
                None
            }
        }
    } else {
        None
    };

    let message = if let Some(fr) = fractal_result {
        format!(
            "File ingested as {} chunks + fractal built ({} parents, {} edges)",
            total_chunks, fr.parent_nodes_created, fr.edges_created
        )
    } else {
        format!("File ingested successfully as {} chunks", total_chunks)
    };

    Ok(Json(IngestResponse {
        success: true,
        node_id: first_node_id,
        embedding_dimension,
        latency_ms: start.elapsed().as_millis() as u64,
        message,
    }))
}

// ============================================================================
// Remember Handler
// ============================================================================

/// Store episodic memory
pub async fn remember(
    State(state): State<SharedState>,
    Json(request): Json<RememberRequest>,
) -> ApiResult<Json<RememberResponse>> {
    if request.content.trim().is_empty() {
        return Err(ApiError::ValidationError("Content cannot be empty".to_string()));
    }

    let state = state.read().await;

    // Generate embedding
    let _ = state
        .brain
        .embed(&request.content)
        .await
        .map_err(|e| ApiError::EmbeddingError(e.to_string()))?;

    // Determine namespace (personal or context-based)
    let namespace = request
        .user_id
        .as_ref()
        .map(|id| format!("user_{}", id))
        .unwrap_or_else(|| "episodic".to_string());

    debug!("Storing episodic memory in namespace: {}", namespace);

    // TODO: Create episodic memory node
    // TODO: Link to related nodes if provided
    // TODO: Link to context if provided

    let node_id = Uuid::new_v4().to_string();

    Ok(Json(RememberResponse {
        success: true,
        node_id: Some(node_id),
        message: "Memory stored successfully".to_string(),
    }))
}

// ============================================================================
// Ask Handler
// ============================================================================

/// Query the knowledge graph with RAG (Retrieval Augmented Generation)
pub async fn ask(
    State(state): State<SharedState>,
    Json(request): Json<AskRequest>,
) -> ApiResult<Json<AskResponse>> {
    let start = Instant::now();

    if request.question.trim().is_empty() {
        return Err(ApiError::ValidationError("Question cannot be empty".to_string()));
    }

    let state = state.read().await;
    let max_results = request.max_results.unwrap_or(5);
    let use_chat = request.use_chat.unwrap_or(true);
    let include_sources = request.include_sources.unwrap_or(true);
    let namespace = request.namespace.as_deref().unwrap_or("global_knowledge");
    let threshold = 0.4; // Lower threshold to get more context

    debug!(
        "Processing question '{}' in namespace '{}', max_results: {}",
        request.question, namespace, max_results
    );

    // 1. Generate embedding for the question
    let query_embedding = state
        .brain
        .embed(&request.question)
        .await
        .map_err(|e| ApiError::EmbeddingError(e.to_string()))?;

    // 2. Search for similar nodes using vector similarity
    let node_repo = NodeRepository::new(&state.db);
    let search_results = node_repo
        .search_similar(&query_embedding.embedding, namespace, max_results * 2)
        .await
        .map_err(|e| ApiError::DatabaseError(format!("Search failed: {}", e)))?;

    // 3. Filter by threshold and check for fractal structure
    let edge_repo = EdgeRepository::new(&state.db);
    let has_fractal = check_fractal_structure(&state.db).await;

    let (filtered_results, used_sssp) = if has_fractal && search_results.len() > 1 {
        // Use SSSP to navigate the fractal graph for better context
        navigate_with_sssp(search_results, threshold, max_results, &node_repo, &edge_repo).await
    } else {
        // Simple vector similarity filtering
        let results: Vec<SearchResult> = search_results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .take(max_results)
            .map(|(node, similarity)| SearchResult {
                node_id: node.id.as_ref().map(|t| t.to_string()).unwrap_or_default(),
                content: node.content,
                similarity,
                namespace: node.namespace,
                node_type: format!("{:?}", node.node_type).to_lowercase(),
                metadata: SearchResultMetadata {
                    source: node.metadata.source,
                    tags: node.metadata.tags,
                },
                depth_level: Some(node.depth_level),
                graph_path: None,
            })
            .collect();
        (results, false)
    };

    debug!(
        "Found {} relevant sources (SSSP: {})",
        filtered_results.len(),
        used_sssp
    );

    // 4. Build context from retrieved sources
    let context = if filtered_results.is_empty() {
        String::new()
    } else {
        filtered_results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let source_info = &r.metadata.source;
                format!(
                    "[Source {}] (from: {}, relevance: {:.1}%)\n{}",
                    i + 1,
                    source_info,
                    r.similarity * 100.0,
                    r.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    };

    // 5. Convert to SourceNode for response
    let sources: Vec<SourceNode> = if include_sources {
        filtered_results
            .iter()
            .map(|r| SourceNode {
                node_id: r.node_id.clone(),
                content: r.content.clone(),
                similarity: r.similarity,
                source: Some(r.metadata.source.clone()),
            })
            .collect()
    } else {
        vec![]
    };

    // 6. Generate response with chat (if enabled)
    let answer = if use_chat {
        let system_prompt = if context.is_empty() {
            "You are a helpful assistant. The user asked a question but no relevant information was found in the knowledge base. \
             Answer based on your general knowledge, but clearly indicate that this is not from the stored knowledge.".to_string()
        } else {
            format!(
                "You are a knowledge assistant that answers questions based ONLY on the provided context.\n\
                 Rules:\n\
                 - Answer using ONLY information from the context below\n\
                 - If the context doesn't contain enough information, say so\n\
                 - Cite which source(s) you used\n\
                 - Be concise and accurate\n\
                 - Answer in the same language as the question\n\n\
                 Context:\n{}\n\n\
                 ---\nNow answer the user's question based on the above context.",
                context
            )
        };

        match state
            .brain
            .chat_with_system(&system_prompt, &request.question)
            .await
        {
            Ok(response) => Some(response.content),
            Err(e) => {
                warn!("Chat generation failed: {}", e);
                Some(format!("Error generating response: {}", e))
            }
        }
    } else {
        // No chat, just return the context
        if context.is_empty() {
            Some("No relevant information found in the knowledge base.".to_string())
        } else {
            Some(format!("Found {} relevant sources:\n\n{}", filtered_results.len(), context))
        }
    };

    info!(
        "Ask completed: {} sources, chat: {}, SSSP: {}, latency: {}ms",
        sources.len(),
        use_chat,
        used_sssp,
        start.elapsed().as_millis()
    );

    Ok(Json(AskResponse {
        success: true,
        answer,
        sources,
        latency_ms: start.elapsed().as_millis() as u64,
        tokens_used: None,
    }))
}

// ============================================================================
// Sync REM Handler
// ============================================================================

/// Trigger REM phase synchronization - consolidates memories and builds fractal hierarchy
pub async fn sync_rem(
    State(state): State<SharedState>,
    Json(request): Json<SyncRemRequest>,
) -> ApiResult<Json<SyncRemResponse>> {
    let start = Instant::now();

    let state = state.read().await;
    let max_nodes = request.max_nodes.unwrap_or(100);
    let enable_clustering = request.enable_clustering.unwrap_or(true);
    let namespace = request.namespace.as_deref().unwrap_or("global_knowledge");

    info!(
        "Starting REM phase sync (namespace: {}, max_nodes: {}, clustering: {})",
        namespace, max_nodes, enable_clustering
    );

    let node_repo = NodeRepository::new(&state.db);
    let mut nodes_processed = 0;
    let mut nodes_created = 0;
    let clusters_formed;

    // 1. Get all leaf nodes in namespace for potential consolidation
    let all_nodes = node_repo
        .get_by_namespace(namespace)
        .await
        .map_err(|e| ApiError::DatabaseError(format!("Failed to list nodes: {}", e)))?;

    let leaf_nodes: Vec<_> = all_nodes
        .into_iter()
        .filter(|n| n.depth_level == 0)
        .take(max_nodes)
        .collect();

    nodes_processed = leaf_nodes.len();
    info!("Found {} leaf nodes to process", nodes_processed);

    // 2. Build fractal hierarchy using RAPTOR if we have enough nodes
    if enable_clustering && leaf_nodes.len() >= 3 {
        info!("Building fractal hierarchy with RAPTOR clustering...");
        
        let config = crate::services::FractalBuilderConfig::new()
            .with_summaries(true)  // Enable LLM summaries for parent nodes
            .with_min_nodes(3);
        
        let fractal_builder = crate::services::FractalBuilder::new(&state.db, config);
        
        match fractal_builder.build_for_namespace(namespace, Some(&state.brain)).await {
            Ok(result) => {
                nodes_created = result.parent_nodes_created;
                clusters_formed = result.edges_created;
                info!(
                    "RAPTOR clustering completed: {} parent nodes, {} edges",
                    result.parent_nodes_created, result.edges_created
                );
            }
            Err(e) => {
                warn!("RAPTOR clustering failed: {}", e);
                clusters_formed = 0;
            }
        }
    } else {
        clusters_formed = 0;
    }

    let elapsed = start.elapsed().as_millis() as u64;
    info!(
        "REM phase completed in {}ms: {} processed, {} created, {} clusters",
        elapsed, nodes_processed, nodes_created, clusters_formed
    );

    Ok(Json(SyncRemResponse {
        success: true,
        nodes_processed,
        nodes_created,
        nodes_updated: 0,
        clusters_formed,
        latency_ms: elapsed,
        message: format!(
            "REM phase completed: {} nodes processed, {} created, {} clusters formed",
            nodes_processed, nodes_created, clusters_formed
        ),
    }))
}

// ============================================================================
// Memory Update Handler
// ============================================================================

/// Update an existing memory node
pub async fn memory_update(
    State(_state): State<SharedState>,
    Json(request): Json<MemoryUpdateRequest>,
) -> ApiResult<Json<MemoryUpdateResponse>> {
    if request.node_id.trim().is_empty() {
        return Err(ApiError::ValidationError("Node ID cannot be empty".to_string()));
    }

    let mut updated_fields = Vec::new();

    // TODO: Fetch existing node from database
    // TODO: Validate node exists

    if request.content.is_some() {
        // TODO: Update content and regenerate embedding
        updated_fields.push("content".to_string());
    }

    if request.status.is_some() {
        // TODO: Update status
        updated_fields.push("status".to_string());
    }

    if request.tags.is_some() {
        // TODO: Update tags
        updated_fields.push("tags".to_string());
    }

    if request.deprecated == Some(true) {
        // TODO: Mark as deprecated
        updated_fields.push("deprecated".to_string());
    }

    // TODO: Save updated node to database

    Ok(Json(MemoryUpdateResponse {
        success: true,
        node_id: request.node_id.clone(),
        updated_fields,
        message: "Memory node updated successfully".to_string(),
    }))
}

// ============================================================================
// Search Handler
// ============================================================================

/// Search for similar content using vector similarity and optional SSSP navigation
pub async fn search(
    State(state): State<SharedState>,
    Json(request): Json<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let start = Instant::now();

    if request.query.trim().is_empty() {
        return Err(ApiError::ValidationError("Query cannot be empty".to_string()));
    }

    let state = state.read().await;
    let limit = request.limit.unwrap_or(10);
    let threshold = request.threshold.unwrap_or(0.5);
    let namespace = request.namespace.as_deref().unwrap_or("global_knowledge");

    debug!(
        "Searching for '{}' in namespace '{}' (limit: {}, threshold: {})",
        request.query, namespace, limit, threshold
    );

    // Generate embedding for the query
    let query_embedding = state
        .brain
        .embed(&request.query)
        .await
        .map_err(|e| ApiError::EmbeddingError(e.to_string()))?;

    // Search using vector similarity
    let node_repo = NodeRepository::new(&state.db);
    let search_results = node_repo
        .search_similar(&query_embedding.embedding, namespace, limit * 2) // Get more for SSSP filtering
        .await
        .map_err(|e| ApiError::DatabaseError(format!("Search failed: {}", e)))?;

    // Check if we have a fractal structure (edges exist)
    let edge_repo = EdgeRepository::new(&state.db);
    let has_fractal_structure = check_fractal_structure(&state.db).await;

    let (results, used_sssp) = if has_fractal_structure && search_results.len() > 1 {
        // Use SSSP to navigate the fractal graph
        navigate_with_sssp(search_results, threshold, limit, &node_repo, &edge_repo).await
    } else {
        // Fall back to simple vector similarity
        let results: Vec<SearchResult> = search_results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .take(limit)
            .map(|(node, similarity)| SearchResult {
                node_id: node.id.as_ref().map(|t| t.to_string()).unwrap_or_default(),
                content: node.content,
                similarity,
                namespace: node.namespace,
                node_type: format!("{:?}", node.node_type).to_lowercase(),
                metadata: SearchResultMetadata {
                    source: node.metadata.source,
                    tags: node.metadata.tags,
                },
                depth_level: Some(node.depth_level),
                graph_path: None,
            })
            .collect();
        (results, false)
    };

    let total = results.len();

    info!(
        "Search completed: found {} results above threshold {} in {}ms (SSSP: {})",
        total,
        threshold,
        start.elapsed().as_millis(),
        used_sssp
    );

    Ok(Json(SearchResponse {
        success: true,
        results,
        total,
        latency_ms: start.elapsed().as_millis() as u64,
        used_sssp: Some(used_sssp),
    }))
}

/// Check if the database has a fractal structure (edges exist)
async fn check_fractal_structure(db: &DatabaseConnection) -> bool {
    let query = "SELECT count() as cnt FROM edges LIMIT 1";
    match db.query(query).await {
        Ok(mut result) => {
            #[derive(serde::Deserialize)]
            struct CountResult { cnt: i64 }
            let counts: Vec<CountResult> = result.take(0).unwrap_or_default();
            counts.first().map(|c| c.cnt > 0).unwrap_or(false)
        }
        Err(_) => false,
    }
}

/// Navigate search results using SSSP algorithm for better relevance
async fn navigate_with_sssp(
    initial_results: Vec<(FractalNode, f32)>,
    threshold: f32,
    limit: usize,
    node_repo: &NodeRepository<'_>,
    edge_repo: &EdgeRepository<'_>,
) -> (Vec<SearchResult>, bool) {
    use crate::graph::{Sssp, GraphNode};
    use std::collections::{HashMap, HashSet};

    // Filter by threshold first
    let filtered: Vec<(FractalNode, f32)> = initial_results
        .into_iter()
        .filter(|(_, sim)| *sim >= threshold)
        .collect();

    if filtered.is_empty() {
        return (vec![], false);
    }

    // Build a local graph from the results and their edges
    let mut graph: HashMap<String, GraphNode> = HashMap::new();
    let mut node_map: HashMap<String, (FractalNode, f32)> = HashMap::new();
    let mut explored_nodes: HashSet<String> = HashSet::new();

    // Add initial nodes to graph
    for (node, similarity) in &filtered {
        let node_id = node.id.as_ref().map(|t| t.to_string()).unwrap_or_default();
        if node_id.is_empty() { continue; }
        
        let graph_node = GraphNode::new(node_id.clone(), node.namespace.clone());
        graph.insert(node_id.clone(), graph_node);
        node_map.insert(node_id.clone(), (node.clone(), *similarity));
        explored_nodes.insert(node_id);
    }

    // Explore parent nodes (go up the fractal hierarchy)
    for (node, _) in &filtered {
        if let Some(node_id) = &node.id {
            // Get incoming edges (from parents)
            if let Ok(incoming) = edge_repo.get_incoming(node_id).await {
                for edge in incoming {
                    let parent_id = edge.from.to_string();
                    if !explored_nodes.contains(&parent_id) {
                        // Fetch parent node
                        if let Ok(Some(parent)) = node_repo.get_by_id(&edge.from).await {
                            let graph_node = GraphNode::new(parent_id.clone(), parent.namespace.clone());
                            graph.insert(parent_id.clone(), graph_node);
                            // Calculate similarity for parent based on edge weight
                            let parent_similarity = edge.similarity * 0.9; // Slight penalty for indirect
                            node_map.insert(parent_id.clone(), (parent, parent_similarity));
                            explored_nodes.insert(parent_id);
                        }
                    }
                }
            }
        }
    }

    // Add edges to graph
    for node_id in explored_nodes.iter() {
        if let Some(thing) = parse_thing_from_string(node_id) {
            if let Ok(outgoing) = edge_repo.get_outgoing(&thing).await {
                for edge in outgoing {
                    let to_id = edge.to.to_string();
                    if explored_nodes.contains(&to_id) {
                        if let Some(gn) = graph.get_mut(node_id) {
                            gn.add_edge(to_id, edge.similarity);
                        }
                    }
                }
            }
        }
    }

    // If we have enough structure, use SSSP to rank
    if graph.len() > 1 {
        let sssp = Sssp::with_defaults();
        
        // Find the best starting node (highest similarity leaf)
        let best_start = filtered.iter()
            .filter(|(n, _)| n.node_type == crate::models::NodeType::Leaf)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(n, _)| n.id.as_ref().map(|t| t.to_string()).unwrap_or_default());

        if let Some(start_id) = best_start {
            if !start_id.is_empty() {
                let sssp_result = sssp.compute(&graph, &start_id, None);
                
                // Combine vector similarity with graph distance for ranking
                let mut ranked: Vec<(String, f32, Option<Vec<String>>)> = node_map
                    .iter()
                    .map(|(id, (_, sim))| {
                        let graph_score = sssp_result.distances.get(id)
                            .map(|&d| 1.0 / (1.0 + d)) // Convert distance to score
                            .unwrap_or(0.0);
                        
                        // Combined score: 70% vector similarity + 30% graph proximity
                        let combined = sim * 0.7 + graph_score * 0.3;
                        
                        // Get path if available
                        let path = sssp_result.reconstruct_path(&start_id, id).map(|p| p.nodes);
                        
                        (id.clone(), combined, path)
                    })
                    .collect();
                
                // Sort by combined score
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                // Convert to results
                let results: Vec<SearchResult> = ranked
                    .into_iter()
                    .take(limit)
                    .filter_map(|(id, _score, path): (String, f32, Option<Vec<String>>)| {
                        node_map.get(&id).map(|(node, orig_sim)| SearchResult {
                            node_id: id.clone(),
                            content: node.content.clone(),
                            similarity: *orig_sim, // Report original similarity
                            namespace: node.namespace.clone(),
                            node_type: format!("{:?}", node.node_type).to_lowercase(),
                            metadata: SearchResultMetadata {
                                source: node.metadata.source.clone(),
                                tags: node.metadata.tags.clone(),
                            },
                            depth_level: Some(node.depth_level),
                            graph_path: path,
                        })
                    })
                    .collect();
                
                return (results, true);
            }
        }
    }

    // Fallback: return simple filtered results
    let results: Vec<SearchResult> = filtered
        .into_iter()
        .take(limit)
        .map(|(node, similarity)| SearchResult {
            node_id: node.id.as_ref().map(|t| t.to_string()).unwrap_or_default(),
            content: node.content,
            similarity,
            namespace: node.namespace,
            node_type: format!("{:?}", node.node_type).to_lowercase(),
            metadata: SearchResultMetadata {
                source: node.metadata.source,
                tags: node.metadata.tags,
            },
            depth_level: Some(node.depth_level),
            graph_path: None,
        })
        .collect();

    (results, false)
}

/// Parse a string ID into a SurrealDB Thing
fn parse_thing_from_string(id: &str) -> Option<surrealdb::sql::Thing> {
    if id.contains(':') {
        let parts: Vec<&str> = id.split(':').collect();
        if parts.len() == 2 {
            Some(surrealdb::sql::Thing::from((parts[0].to_string(), parts[1].to_string())))
        } else {
            None
        }
    } else {
        Some(surrealdb::sql::Thing::from(("nodes".to_string(), id.to_string())))
    }
}

// ============================================================================
// Build Fractal Handler
// ============================================================================

/// Build fractal hierarchical structure for a namespace
pub async fn build_fractal(
    State(state): State<SharedState>,
    Json(request): Json<BuildFractalRequest>,
) -> ApiResult<Json<BuildFractalResponse>> {
    use crate::services::{FractalBuilder, FractalBuilderConfig};
    use crate::graph::RaptorConfig;

    let start = Instant::now();
    let state = state.read().await;
    let namespace = request.namespace.as_deref().unwrap_or("global_knowledge");

    info!("Building fractal structure for namespace '{}'", namespace);

    // Configure RAPTOR
    let mut raptor_config = RaptorConfig::default();
    if let Some(threshold) = request.similarity_threshold {
        raptor_config = raptor_config.with_similarity_threshold(threshold);
    }
    if let Some(max_depth) = request.max_depth {
        raptor_config = raptor_config.with_max_depth(max_depth);
    }

    // Configure fractal builder
    let builder_config = FractalBuilderConfig::new()
        .with_raptor_config(raptor_config)
        .with_summaries(request.generate_summaries.unwrap_or(true));

    let builder = FractalBuilder::new(&state.db, builder_config);

    // Build the fractal structure
    let brain_ref: Option<&crate::models::llm::ModelBrain> = Some(&state.brain);

    let result = builder
        .build_for_namespace(namespace, brain_ref)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to build fractal: {}", e)))?;

    let response = BuildFractalResponse {
        success: true,
        parent_nodes_created: result.parent_nodes_created,
        edges_created: result.edges_created,
        max_depth: result.max_depth,
        root_node_ids: result.root_node_ids,
        latency_ms: start.elapsed().as_millis() as u64,
        message: format!(
            "Fractal structure built: {} parent nodes, {} edges, max depth {}",
            result.parent_nodes_created,
            result.edges_created,
            result.max_depth
        ),
    };

    info!(
        "Fractal build completed in {}ms: {} parent nodes, {} edges",
        response.latency_ms,
        response.parent_nodes_created,
        response.edges_created
    );

    Ok(Json(response))
}

// ============================================================================
// Stats Handler
// ============================================================================

/// Get system statistics
pub async fn stats(State(state): State<SharedState>) -> Json<StatsResponse> {
    let state = state.read().await;

    let cache_metrics = state.node_cache.metrics();
    let llm_info = state.brain.get_models_info();

    Json(StatsResponse {
        total_nodes: 0,    // TODO: Get from database
        total_edges: 0,    // TODO: Get from database
        namespaces: vec![], // TODO: Get namespace stats
        cache_metrics: CacheStats {
            size: cache_metrics.size,
            capacity: state.node_cache.capacity(),
            hit_rate: cache_metrics.hit_rate(),
            hits: cache_metrics.hits,
            misses: cache_metrics.misses,
        },
        llm_metrics: LlmStats {
            embedding_model: llm_info.embedding_model,
            chat_model: llm_info.chat_model,
            summarizer_model: llm_info.summarizer_model,
            is_local: state.brain.is_fully_local(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_type() {
        // Just verify the types compile correctly
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedState>();
    }

    #[test]
    fn test_parse_tags_json_or_plain() {
        let json = r#"["tag1","tag2"]"#;
        let parsed: Vec<String> = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, vec!["tag1".to_string(), "tag2".to_string()]);

        let plain = "single-tag";
        // parsing plain as JSON should fail, so we treat it as single tag
        assert!(serde_json::from_str::<Vec<String>>(plain).is_err());
    }
}

// ============================================================================
// Model Upload Handlers
// ============================================================================

use axum::extract::Path;
use axum::body::Bytes;

/// Initialize a chunked upload session
pub async fn init_model_upload(
    State(state): State<SharedState>,
    Json(request): Json<InitUploadRequest>,
) -> ApiResult<Json<InitUploadResponse>> {
    // Validate request
    if !request.filename.to_lowercase().ends_with(".gguf") {
        return Err(ApiError::ValidationError(
            "File must be a .gguf model file".to_string(),
        ));
    }
    
    if request.total_size == 0 {
        return Err(ApiError::ValidationError(
            "File size cannot be 0".to_string(),
        ));
    }
    
    // Max 500GB
    const MAX_SIZE: u64 = 500 * 1024 * 1024 * 1024;
    if request.total_size > MAX_SIZE {
        return Err(ApiError::ValidationError(format!(
            "File size {} exceeds maximum allowed {} bytes (500GB)",
            request.total_size, MAX_SIZE
        )));
    }
    
    // Validate chunk size (10MB - 500MB)
    const MIN_CHUNK: u64 = 10 * 1024 * 1024;
    const MAX_CHUNK: u64 = 500 * 1024 * 1024;
    let chunk_size = request.chunk_size.clamp(MIN_CHUNK, MAX_CHUNK);
    
    let state_read = state.read().await;
    let manager = &state_read.upload_manager;
    
    let session = manager
        .init_upload(request.filename, request.total_size, Some(chunk_size))
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to init upload: {}", e)))?;
    
    info!(
        "Initialized model upload: id={}, chunks={}",
        session.upload_id, session.total_chunks
    );
    
    Ok(Json(InitUploadResponse {
        upload_id: session.upload_id,
        chunk_size: session.chunk_size,
        total_chunks: session.total_chunks,
    }))
}

/// Query parameters for chunk upload
#[derive(Debug, Deserialize)]
pub struct ChunkParams {
    pub chunk_index: u64,
    pub checksum: Option<String>,
}

/// Upload a chunk of the model file
pub async fn upload_model_chunk(
    State(state): State<SharedState>,
    Path(upload_id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<ChunkParams>,
    body: Bytes,
) -> ApiResult<Json<UploadChunkResponse>> {
    let state_read = state.read().await;
    let manager = &state_read.upload_manager;
    
    let result = manager
        .upload_chunk(
            &upload_id,
            params.chunk_index,
            body,
            params.checksum.as_deref(),
        )
        .await
        .map_err(|e| ApiError::BadRequest(format!("Chunk upload failed: {}", e)))?;
    
    debug!(
        "Received chunk {} for upload {} ({}/{})",
        params.chunk_index, upload_id, result.chunks_received, result.total_chunks
    );
    
    Ok(Json(UploadChunkResponse {
        success: result.success,
        chunk_index: result.chunk_index,
        chunks_received: result.chunks_received,
        total_chunks: result.total_chunks,
    }))
}

/// Finalize an upload after all chunks received
pub async fn finalize_model_upload(
    State(state): State<SharedState>,
    Path(upload_id): Path<String>,
) -> ApiResult<Json<FinalizeUploadResponse>> {
    let state_read = state.read().await;
    let manager = &state_read.upload_manager;
    
    // Get session info before finalizing
    let session = manager
        .get_status(&upload_id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Upload session not found: {}", upload_id)))?;
    
    let filename = session.filename.clone();
    let total_size = session.total_size;
    
    // Finalize the upload (move file to final location)
    let result = manager
        .finalize(&upload_id)
        .await
        .map_err(|e| ApiError::BadRequest(format!("Finalize failed: {}", e)))?;
    
    // Create the FractalModel record in the database
    let model = crate::models::llm::fractal_model::FractalModel::new(
        filename.clone(),
        result.file_path.clone(),
        total_size,
    );
    
    let repo = crate::db::queries::FractalModelRepository::new(&state_read.db);
    let model_id = repo
        .create(&model)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to create model record: {}", e)))?;
    
    info!(
        "Finalized model upload: upload_id={}, model_id={}, file={}, size={}",
        upload_id, model_id, result.file_path, total_size
    );
    
    // Auto-start conversion in background
    let db_clone = state_read.db.clone();
    let model_id_clone = model_id.clone();
    let file_path = result.file_path.clone();
    let upload_manager_clone = state_read.upload_manager.clone();
    let upload_id_clone = upload_id.clone();
    
    tokio::spawn(async move {
        info!("Auto-starting conversion for model: {}", model_id_clone);
        
        // Update status to converting
        let repo = crate::db::queries::FractalModelRepository::new(&db_clone);
        if let Err(e) = repo.update_status(&model_id_clone, crate::models::llm::fractal_model::FractalModelStatus::Converting).await {
            error!("Failed to update model status: {}", e);
            return;
        }
        
        // Run conversion
        if let Err(e) = run_model_conversion(&db_clone, &model_id_clone, &file_path).await {
            error!("Model conversion failed for {}: {}", model_id_clone, e);
            let _ = repo.update_status(&model_id_clone, crate::models::llm::fractal_model::FractalModelStatus::Failed).await;
            let _ = upload_manager_clone.mark_failed(&upload_id_clone, &e.to_string()).await;
            return;
        }
        
        // Mark upload session as ready so frontend stops polling
        if let Err(e) = upload_manager_clone.mark_ready(&upload_id_clone).await {
            error!("Failed to mark upload as ready: {}", e);
        }
    });
    
    Ok(Json(FinalizeUploadResponse {
        success: result.success,
        model_id,
        message: "Upload finalized. Conversion started automatically.".to_string(),
    }))
}

/// Get current upload status
pub async fn get_upload_status(
    State(state): State<SharedState>,
    Path(upload_id): Path<String>,
) -> ApiResult<Json<ProgressResponse>> {
    let state_read = state.read().await;
    let manager = &state_read.upload_manager;
    
    let session = manager
        .get_status(&upload_id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Upload session not found: {}", upload_id)))?;
    
    Ok(Json(ProgressResponse {
        upload_progress: session.upload_progress,
        conversion_progress: session.conversion_progress,
        status: session.status.as_str().to_string(),
        upload_speed_mbps: session.upload_speed_mbps,
        chunks_received: Some(session.chunks_received.len() as u64),
        total_chunks: Some(session.total_chunks),
        current_phase: session.current_phase,
    }))
}

/// Cancel an ongoing upload
pub async fn cancel_model_upload(
    State(state): State<SharedState>,
    Path(upload_id): Path<String>,
) -> ApiResult<Json<CancelUploadResponse>> {
    let state_read = state.read().await;
    let manager = &state_read.upload_manager;
    
    manager
        .cancel(&upload_id)
        .await
        .map_err(|e| ApiError::InternalError(format!("Cancel failed: {}", e)))?;
    
    info!("Cancelled model upload: {}", upload_id);
    
    Ok(Json(CancelUploadResponse {
        success: true,
        message: "Upload cancelled and temporary files cleaned up".to_string(),
    }))
}

/// SSE stream for upload/conversion progress
pub async fn upload_progress_stream(
    State(state): State<SharedState>,
    Path(upload_id): Path<String>,
) -> impl axum::response::IntoResponse {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use std::convert::Infallible;
    
    // Clone the Arc to move into the async stream
    let manager = state.read().await.upload_manager.clone();
    
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
        
        loop {
            interval.tick().await;
            
            let session = manager.get_status(&upload_id).await;
            
            match session {
                Some(s) => {
                    let progress = ProgressResponse {
                        upload_progress: s.upload_progress,
                        conversion_progress: s.conversion_progress,
                        status: s.status.as_str().to_string(),
                        upload_speed_mbps: s.upload_speed_mbps,
                        chunks_received: Some(s.chunks_received.len() as u64),
                        total_chunks: Some(s.total_chunks),
                        current_phase: s.current_phase.clone(),
                    };
                    
                    let json = serde_json::to_string(&progress).unwrap_or_default();
                    yield Ok::<_, Infallible>(Event::default().data(json));
                    
                    // Stop when ready or failed
                    if s.status == crate::models::upload_session::UploadStatus::Ready 
                        || s.status == crate::models::upload_session::UploadStatus::Failed 
                    {
                        break;
                    }
                }
                None => {
                    let error = serde_json::json!({
                        "error": "Session not found",
                        "upload_id": upload_id
                    });
                    yield Ok::<_, Infallible>(Event::default().event("error").data(error.to_string()));
                    break;
                }
            }
        }
    };
    
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ============================================================================
// Model Management Handlers
// ============================================================================

use crate::db::queries::FractalModelRepository;
use crate::models::llm::fractal_model::FractalModelStatus;

/// List available Ollama models
pub async fn list_ollama_models() -> ApiResult<Json<ListOllamaModelsResponse>> {
    let ollama_base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    
    let client = reqwest::Client::new();
    let url = format!("{}/api/tags", ollama_base_url);
    
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to connect to Ollama: {}", e)))?;
    
    if !response.status().is_success() {
        return Err(ApiError::InternalError(format!(
            "Ollama returned error: {}",
            response.status()
        )));
    }
    
    #[derive(Deserialize)]
    struct OllamaTagsResponse {
        models: Vec<OllamaTagModel>,
    }
    
    #[derive(Deserialize)]
    struct OllamaTagModel {
        name: String,
        model: Option<String>,
        modified_at: String,
        size: u64,
        digest: String,
        details: Option<OllamaTagDetails>,
    }
    
    #[derive(Deserialize)]
    struct OllamaTagDetails {
        parent_model: Option<String>,
        format: Option<String>,
        family: Option<String>,
        families: Option<Vec<String>>,
        parameter_size: Option<String>,
        quantization_level: Option<String>,
    }
    
    let ollama_response: OllamaTagsResponse = response
        .json()
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to parse Ollama response: {}", e)))?;
    
    let models: Vec<OllamaModelInfo> = ollama_response
        .models
        .into_iter()
        .map(|m| OllamaModelInfo {
            name: m.name.clone(),
            model: m.model.unwrap_or_else(|| m.name.clone()),
            modified_at: m.modified_at,
            size: m.size,
            digest: m.digest,
            details: m.details.map(|d| OllamaModelDetails {
                parent_model: d.parent_model,
                format: d.format,
                family: d.family,
                families: d.families,
                parameter_size: d.parameter_size,
                quantization_level: d.quantization_level,
            }),
        })
        .collect();
    
    Ok(Json(ListOllamaModelsResponse { models }))
}

/// List all fractal models
pub async fn list_models(
    State(state): State<SharedState>,
) -> ApiResult<Json<ListModelsResponse>> {
    let state_read = state.read().await;
    let repo = FractalModelRepository::new(&state_read.db);
    
    let models = repo
        .list_all()
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to list models: {}", e)))?;
    
    let model_infos: Vec<ModelInfo> = models
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            name: m.name,
            status: match m.status {
                FractalModelStatus::Uploading => "uploading".to_string(),
                FractalModelStatus::Uploaded => "uploaded".to_string(),
                FractalModelStatus::Converting => "converting".to_string(),
                FractalModelStatus::Ready => "ready".to_string(),
                FractalModelStatus::Failed => "failed".to_string(),
            },
            architecture: Some(serde_json::to_value(&m.architecture).unwrap_or_default()),
            file_size: m.file_size,
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();
    
    Ok(Json(ListModelsResponse {
        models: model_infos,
    }))
}

/// Get details of a specific model
pub async fn get_model(
    State(state): State<SharedState>,
    Path(model_id): Path<String>,
) -> ApiResult<Json<GetModelResponse>> {
    let state_read = state.read().await;
    let repo = FractalModelRepository::new(&state_read.db);
    
    let model = repo
        .get_by_id(&model_id)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to get model: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;
    
    Ok(Json(GetModelResponse {
        model: ModelInfo {
            id: model.id,
            name: model.name,
            status: match model.status {
                FractalModelStatus::Uploading => "uploading".to_string(),
                FractalModelStatus::Uploaded => "uploaded".to_string(),
                FractalModelStatus::Converting => "converting".to_string(),
                FractalModelStatus::Ready => "ready".to_string(),
                FractalModelStatus::Failed => "failed".to_string(),
            },
            architecture: Some(serde_json::to_value(&model.architecture).unwrap_or_default()),
            file_size: model.file_size,
            created_at: model.created_at.to_rfc3339(),
        },
    }))
}

/// Delete a model
pub async fn delete_model(
    State(state): State<SharedState>,
    Path(model_id): Path<String>,
) -> ApiResult<Json<DeleteModelResponse>> {
    let state_read = state.read().await;
    let repo = FractalModelRepository::new(&state_read.db);
    
    // Get the model first to get the file path
    let model = repo
        .get_by_id(&model_id)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to get model: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;
    
    // Delete from database (also deletes associated nodes)
    repo.delete(&model_id)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to delete model: {}", e)))?;
    
    // Try to delete the file (don't fail if file doesn't exist)
    if let Err(e) = tokio::fs::remove_file(&model.file_path).await {
        warn!("Failed to delete model file {}: {}", model.file_path, e);
    }
    
    info!("Deleted model: {} ({})", model_id, model.name);
    
    Ok(Json(DeleteModelResponse {
        success: true,
        message: format!("Model {} deleted successfully", model_id),
    }))
}

/// Start model conversion (GGUF -> Fractal)
pub async fn convert_model(
    State(state): State<SharedState>,
    Path(model_id): Path<String>,
) -> ApiResult<Json<ConvertModelResponse>> {
    let state_read = state.read().await;
    let repo = FractalModelRepository::new(&state_read.db);
    
    // Get the model
    let model = repo
        .get_by_id(&model_id)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to get model: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;
    
    // Check if already converting or ready
    match model.status {
        FractalModelStatus::Converting => {
            return Err(ApiError::BadRequest("Model is already being converted".to_string()));
        }
        FractalModelStatus::Ready => {
            return Err(ApiError::BadRequest("Model is already converted and ready".to_string()));
        }
        _ => {}
    }
    
    // Update status to converting
    repo.update_status(&model_id, FractalModelStatus::Converting)
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to update model status: {}", e)))?;
    
    // Spawn async conversion task
    let db_clone = state_read.db.clone();
    let model_id_clone = model_id.clone();
    let file_path = model.file_path.clone();
    
    tokio::spawn(async move {
        if let Err(e) = run_model_conversion(&db_clone, &model_id_clone, &file_path).await {
            error!("Model conversion failed for {}: {}", model_id_clone, e);
            let repo = FractalModelRepository::new(&db_clone);
            let _ = repo.update_status(&model_id_clone, FractalModelStatus::Failed).await;
        }
    });
    
    info!("Started conversion for model: {} ({})", model_id, model.name);
    
    Ok(Json(ConvertModelResponse {
        success: true,
        message: format!("Conversion started for model {}. Monitor progress via /v1/models/{}/status", model_id, model_id),
    }))
}

// ============================================================================
// Model Conversion Logic
// ============================================================================

use crate::services::ModelConversionService;

/// Run the model conversion process using the real conversion service
async fn run_model_conversion(
    db: &DatabaseConnection,
    model_id: &str,
    _file_path: &str, // No longer needed, service gets it from model
) -> Result<(), anyhow::Error> {
    let repo = FractalModelRepository::new(db);
    
    // Get the model from database
    let mut model = repo
        .get_by_id(model_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;
    
    // Create conversion service and run real conversion
    let conversion_service = ModelConversionService::new(std::sync::Arc::new(db.clone()));
    conversion_service.convert_model(&mut model).await?;
    
    Ok(())
}
