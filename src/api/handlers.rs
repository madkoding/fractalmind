//! API request handlers.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use axum::{extract::{State, Multipart}, Json};
use serde_json;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::services::ingestion::extractors::{ExtractorFactory, ContentExtractor};
use crate::services::ingestion::config::{FileType, IngestionConfig};
use crate::db::queries::NodeRepository;

use crate::cache::{EmbeddingCache, NodeCache};
use crate::db::connection::DatabaseConnection;
use crate::models::llm::ModelBrain;
use crate::models::{EmbeddingVector, FractalNode, NodeMetadata};

use super::error::{ApiError, ApiResult};
use super::types::*;

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

    // Create fractal node
    let _node = FractalNode::new_leaf(
        request.content.clone(),
        embedding_vector,
        namespace,
        request.source.clone(),
        metadata,
    );

    // TODO: Save node to database
    // let node_repo = NodeRepository::new(&state.db);
    // let node_id = node_repo.create(&node).await?;

    let node_id = Uuid::new_v4().to_string();

    Ok(Json(IngestResponse {
        success: true,
        node_id: Some(node_id),
        embedding_dimension: Some(embedding_response.dimension),
        latency_ms: start.elapsed().as_millis() as u64,
        message: "Content ingested successfully".to_string(),
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

    // Generate embedding (allow tests to disable real model calls)
    let state = state.read().await;

    let disable_embedding = std::env::var("TEST_DISABLE_EMBEDDING")
        .map(|v| v == "true")
        .unwrap_or(false);

    let embedding_response = if disable_embedding {
        // Build a fake embedding response using EMBEDDING_DIMENSION env var
        let dim = std::env::var("EMBEDDING_DIMENSION")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(768);
        crate::models::EmbeddingVector::new(vec![0.0f32; dim], crate::models::EmbeddingModel::NomicEmbedTextV15);

        // Create a slice and mimic EmbeddingResponse fields inline
        crate::models::EmbeddingVector::new(vec![0.0f32; dim], crate::models::EmbeddingModel::NomicEmbedTextV15);

        // We'll construct the EmbeddingResponse-like values manually below
        let embedding_vec = vec![0.0f32; dim];
        crate::models::llm::traits_llm::EmbeddingResponse {
            embedding: embedding_vec,
            dimension: dim,
            model: "test-mock".to_string(),
            latency_ms: 0,
        }
    } else {
        state
            .brain
            .embed(&extraction.text)
            .await
            .map_err(|e| ApiError::EmbeddingError(e.to_string()))?
    };

    // Cache embedding
    let embedding_vector = EmbeddingVector::new(
        embedding_response.embedding.clone(),
        crate::models::EmbeddingModel::NomicEmbedTextV15,
    );
    state.embedding_cache.put(&extraction.text, embedding_vector.clone());

    // Prepare metadata
    let mut metadata = NodeMetadata::default();
    metadata.source = filename.clone().unwrap_or_else(|| "uploaded_file".to_string());
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
        .or(language)
        .unwrap_or_else(|| "en".to_string());
    if let Some(t) = &tags {
        metadata.tags = t.clone();
    }

    // Create node
    let node = FractalNode::new_leaf(
        extraction.text.clone(),
        embedding_vector,
        namespace.clone(),
        filename.clone(),
        metadata,
    );

    // Persist (allow tests to skip actual DB writes)
    let skip_db = std::env::var("TEST_SKIP_DB_WRITES").map(|v| v == "true").unwrap_or(false);
    let node_id = if skip_db {
        // Generate a fake node id (UUID) when DB writes are disabled for tests
        Uuid::new_v4().to_string()
    } else {
        let node_repo = NodeRepository::new(&state.db);
        let created = node_repo
            .create(&node)
            .await
            .map_err(|e| ApiError::DatabaseError(format!("Failed to create node: {}", e)))?;
        created.to_string()
    };

    Ok(Json(IngestResponse {
        success: true,
        node_id: Some(node_id),
        embedding_dimension: Some(embedding_response.dimension),
        latency_ms: start.elapsed().as_millis() as u64,
        message: "File ingested successfully".to_string(),
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

/// Query the knowledge graph
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

    debug!(
        "Processing question in namespace: {:?}, max_results: {}",
        request.namespace, max_results
    );

    // 1. Generate embedding for the question
    let _query_embedding = state
        .brain
        .embed(&request.question)
        .await
        .map_err(|e| ApiError::EmbeddingError(e.to_string()))?;

    // TODO: 2. Search for similar nodes using HNSW index
    // let similar_nodes = node_repo.search_similar(
    //     &query_embedding.embedding,
    //     namespace,
    //     max_results,
    // ).await?;

    // TODO: 3. Navigate the fractal graph using SSSP with hopsets
    // TODO: 4. Collect context from relevant nodes

    // Mock sources for now
    let sources: Vec<SourceNode> = vec![];

    // 5. Generate response with chat (if enabled)
    let answer = if use_chat {
        let system_prompt = "You are a knowledge assistant. Answer the following question based on the provided context.\n\
             If you don't have enough information, say so honestly.\n\
             Context: (No relevant context found yet)".to_string();

        match state
            .brain
            .chat_with_system(&system_prompt, &request.question)
            .await
        {
            Ok(response) => Some(response.content),
            Err(e) => {
                warn!("Chat generation failed: {}", e);
                None
            }
        }
    } else {
        None
    };

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

/// Trigger REM phase synchronization
pub async fn sync_rem(
    State(state): State<SharedState>,
    Json(request): Json<SyncRemRequest>,
) -> ApiResult<Json<SyncRemResponse>> {
    let start = Instant::now();

    let state = state.read().await;
    let max_nodes = request.max_nodes.unwrap_or(100);
    let enable_clustering = request.enable_clustering.unwrap_or(true);

    info!(
        "Starting REM phase sync (max_nodes: {}, clustering: {})",
        max_nodes, enable_clustering
    );

    // TODO: 1. Get nodes with status "incomplete"
    // let incomplete_nodes = node_repo.get_by_status(NodeStatus::Incomplete).await?;

    // TODO: 2. For each incomplete node, optionally perform web search
    // if request.enable_web_search.unwrap_or(false) { ... }

    // TODO: 3. Summarize gathered information
    // TODO: 4. Create new synthesized nodes

    // TODO: 5. Perform RAPTOR clustering if enabled
    // if enable_clustering { ... }

    // For now, verify the summarizer works
    let test_text = "This is a test of the REM phase summarization capability.";
    match state.brain.summarize(test_text).await {
        Ok(summary) => {
            debug!("REM phase summarizer test passed: {}", summary);
        }
        Err(e) => {
            error!("REM phase summarizer failed: {}", e);
            return Err(ApiError::LlmError(format!("Summarizer failed: {}", e)));
        }
    }

    Ok(Json(SyncRemResponse {
        success: true,
        nodes_processed: 0,
        nodes_created: 0,
        nodes_updated: 0,
        clusters_formed: 0,
        latency_ms: start.elapsed().as_millis() as u64,
        message: "REM phase sync completed (development mode)".to_string(),
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

/// Search for similar content
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

    debug!(
        "Searching for '{}' (limit: {}, threshold: {})",
        request.query, limit, threshold
    );

    // Generate embedding for the query
    let _ = state
        .brain
        .embed(&request.query)
        .await
        .map_err(|e| ApiError::EmbeddingError(e.to_string()))?;

    // TODO: Search using HNSW index
    // let results = node_repo.search_similar(&embedding, namespace, limit).await?;

    // Filter by threshold
    // let results: Vec<_> = results.into_iter()
    //     .filter(|(_, similarity)| *similarity >= threshold)
    //     .collect();

    Ok(Json(SearchResponse {
        success: true,
        results: vec![],
        total: 0,
        latency_ms: start.elapsed().as_millis() as u64,
    }))
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

// ============================================================================
// Model Management Handlers
// ============================================================================

/// Upload a GGUF model file
pub async fn upload_model(
    State(state): State<SharedState>,
    mut multipart: Multipart,
) -> ApiResult<Json<UploadModelResponse>> {
    use tokio::fs;
    use tokio::io::AsyncWriteExt;
    use crate::models::llm::FractalModel;
    use crate::services::ModelConversionService;
    use std::sync::Arc;
    
    const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB
    const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" en little-endian
    
    let state = state.read().await;
    
    // Crear directorio para modelos si no existe
    let models_dir = "/var/tmp/fractalmind_models";
    fs::create_dir_all(models_dir).await?;
    
    let mut file_name = String::new();
    let mut file_path = String::new();
    let mut file_size = 0u64;
    
    // Procesar multipart upload
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "file" {
            file_name = field.file_name().unwrap_or("model.gguf").to_string();
            
            // Validar extensión
            if !file_name.to_lowercase().ends_with(".gguf") {
                return Err(ApiError::ValidationError(
                    "Only .gguf files are supported".to_string()
                ));
            }
            
            file_path = format!("{}/{}", models_dir, file_name);
            
            // Escribir archivo
            let data = field.bytes().await?;
            file_size = data.len() as u64;
            
            // Validar tamaño
            if file_size > MAX_FILE_SIZE {
                return Err(ApiError::ValidationError(
                    format!("File too large: {} bytes (max: {} bytes)", file_size, MAX_FILE_SIZE)
                ));
            }
            
            if file_size < 8 {
                return Err(ApiError::ValidationError(
                    "File too small to be a valid GGUF model".to_string()
                ));
            }
            
            // Validar magic number GGUF
            if data.len() >= 4 {
                let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                if magic != GGUF_MAGIC {
                    return Err(ApiError::ValidationError(
                        format!("Invalid GGUF file: magic number mismatch (expected 0x{:08x}, got 0x{:08x})", 
                            GGUF_MAGIC, magic)
                    ));
                }
            }
            
            let mut file = fs::File::create(&file_path).await?;
            file.write_all(&data).await?;
        }
    }
    
    if file_path.is_empty() {
        return Err(ApiError::ValidationError("No file uploaded".to_string()));
    }
    
    // Crear modelo en DB
    let model = FractalModel::new(file_name.clone(), file_path, file_size);
    let model_id = model.id.clone();
    
    let conversion_service = ModelConversionService::new(Arc::new(state.db.clone()));
    conversion_service.create_model(&model).await?;
    
    Ok(Json(UploadModelResponse {
        success: true,
        model_id,
        message: format!("Model {} uploaded successfully", file_name),
    }))
}

/// Convert a model to fractal structure
pub async fn convert_model(
    State(state): State<SharedState>,
    Json(request): Json<ConvertModelRequest>,
) -> ApiResult<Json<ConvertModelResponse>> {
    use crate::services::ModelConversionService;
    use std::sync::Arc;
    
    let state = state.read().await;
    let conversion_service = ModelConversionService::new(Arc::new(state.db.clone()));
    
    // Obtener modelo
    let mut model = conversion_service
        .get_model(&request.model_id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", request.model_id)))?;
    
    // Iniciar conversión en background
    let db = Arc::new(state.db.clone());
    let model_id = model.id.clone();
    
    tokio::spawn(async move {
        let service = ModelConversionService::new(db);
        if let Err(e) = service.convert_model(&mut model).await {
            error!("Failed to convert model {}: {}", model_id, e);
        }
    });
    
    Ok(Json(ConvertModelResponse {
        success: true,
        message: format!("Model conversion started for {}", request.model_id),
    }))
}

/// List all models
pub async fn list_models(
    State(state): State<SharedState>,
) -> ApiResult<Json<ListModelsResponse>> {
    use crate::services::ModelConversionService;
    use std::sync::Arc;
    
    let state = state.read().await;
    let conversion_service = ModelConversionService::new(Arc::new(state.db.clone()));
    
    let models = conversion_service.list_models().await?;
    
    let model_infos: Vec<ModelInfo> = models
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            name: m.name,
            status: format!("{:?}", m.status).to_lowercase(),
            architecture: serde_json::to_value(&m.architecture).ok(),
            file_size: m.file_size,
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();
    
    Ok(Json(ListModelsResponse {
        models: model_infos,
    }))
}

/// Get model details
pub async fn get_model(
    State(state): State<SharedState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> ApiResult<Json<GetModelResponse>> {
    use crate::services::ModelConversionService;
    use std::sync::Arc;
    
    let state = state.read().await;
    let conversion_service = ModelConversionService::new(Arc::new(state.db.clone()));
    
    let model = conversion_service
        .get_model(&model_id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;
    
    Ok(Json(GetModelResponse {
        model: ModelInfo {
            id: model.id,
            name: model.name,
            status: format!("{:?}", model.status).to_lowercase(),
            architecture: serde_json::to_value(&model.architecture).ok(),
            file_size: model.file_size,
            created_at: model.created_at.to_rfc3339(),
        },
    }))
}

/// Delete a model
pub async fn delete_model(
    State(state): State<SharedState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> ApiResult<Json<DeleteModelResponse>> {
    use crate::services::ModelConversionService;
    use std::sync::Arc;
    
    let state = state.read().await;
    let conversion_service = ModelConversionService::new(Arc::new(state.db.clone()));
    
    conversion_service.delete_model(&model_id).await?;
    
    Ok(Json(DeleteModelResponse {
        success: true,
        message: format!("Model {} deleted successfully", model_id),
    }))
}

/// Update model strategy (fractal vs ollama)
pub async fn update_strategy(
    State(_state): State<SharedState>,
    Json(request): Json<UpdateStrategyRequest>,
) -> ApiResult<Json<UpdateStrategyResponse>> {
    // TODO: Implementar storage de configuración de estrategia
    // Por ahora, solo validamos y retornamos
    
    match request.strategy.as_str() {
        "fractal" => {
            if request.model_id.is_none() {
                return Err(ApiError::ValidationError(
                    "model_id required for fractal strategy".to_string(),
                ));
            }
        }
        "ollama" => {}
        _ => {
            return Err(ApiError::ValidationError(format!(
                "Invalid strategy: {}. Must be 'fractal' or 'ollama'",
                request.strategy
            )));
        }
    }
    
    Ok(Json(UpdateStrategyResponse {
        success: true,
        current_strategy: request.strategy.clone(),
        message: format!("Strategy updated to {}", request.strategy),
    }))
}

/// List available Ollama models
pub async fn list_ollama_models() -> ApiResult<Json<ListOllamaModelsResponse>> {
    use reqwest::Client;
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    struct OllamaApiModel {
        name: String,
        model: String,
        modified_at: String,
        size: u64,
        digest: String,
        details: Option<OllamaApiDetails>,
    }
    
    #[derive(Deserialize)]
    struct OllamaApiDetails {
        parent_model: Option<String>,
        format: Option<String>,
        family: Option<String>,
        families: Option<Vec<String>>,
        parameter_size: Option<String>,
        quantization_level: Option<String>,
    }
    
    #[derive(Deserialize)]
    struct OllamaTagsResponse {
        models: Vec<OllamaApiModel>,
    }
    
    let ollama_url = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    
    let client = Client::new();
    let url = format!("{}/api/tags", ollama_url);
    
    debug!("Fetching Ollama models from: {}", url);
    
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to connect to Ollama: {}", e)))?;
    
    if !response.status().is_success() {
        return Err(ApiError::InternalError(format!(
            "Ollama API returned status: {}",
            response.status()
        )));
    }
    
    let ollama_response: OllamaTagsResponse = response
        .json()
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to parse Ollama response: {}", e)))?;
    
    let models = ollama_response
        .models
        .into_iter()
        .map(|m| OllamaModelInfo {
            name: m.name,
            model: m.model,
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
