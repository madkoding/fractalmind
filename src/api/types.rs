//! API request/response types.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ============================================================================
// Health Check
// ============================================================================

/// Health check response
#[derive(Serialize, Clone)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub components: HealthComponents,
}

/// Health status of individual components
#[derive(Serialize, Clone)]
pub struct HealthComponents {
    pub database: bool,
    pub llm: bool,
    pub cache: bool,
}

// ============================================================================
// Ingest API
// ============================================================================

/// Request to ingest content into the knowledge graph
#[derive(Deserialize)]
pub struct IngestRequest {
    /// The content to ingest
    pub content: String,

    /// Source of the content (e.g., URL, file path)
    pub source: Option<String>,

    /// Type of source (e.g., "pdf", "web", "text", "image")
    pub source_type: Option<String>,

    /// Namespace to store the content in
    pub namespace: Option<String>,

    /// Tags for categorization
    pub tags: Option<Vec<String>>,

    /// Language of the content
    pub language: Option<String>,
}

/// Response from ingest operation
#[derive(Serialize)]
pub struct IngestResponse {
    pub success: bool,
    pub node_id: Option<String>,
    pub embedding_dimension: Option<usize>,
    pub latency_ms: u64,
    pub message: String,
}

// ============================================================================
// Remember API (Episodic Memory)
// ============================================================================

/// Request to store episodic memory
#[derive(Deserialize)]
pub struct RememberRequest {
    /// The content to remember
    pub content: String,

    /// Conversation context
    pub context: Option<String>,

    /// Related node IDs
    pub related_nodes: Option<Vec<String>>,

    /// User identifier for personal namespace
    pub user_id: Option<String>,
}

/// Response from remember operation
#[derive(Serialize)]
pub struct RememberResponse {
    pub success: bool,
    pub node_id: Option<String>,
    pub message: String,
}

// ============================================================================
// Ask API (Knowledge Query)
// ============================================================================

/// Request to query the knowledge graph
#[derive(Deserialize)]
pub struct AskRequest {
    /// The question to ask
    pub question: String,

    /// Namespace to search in
    pub namespace: Option<String>,

    /// Maximum number of source nodes to consider
    pub max_results: Option<usize>,

    /// Whether to include source nodes in response
    pub include_sources: Option<bool>,

    /// Whether to use chat for response generation
    pub use_chat: Option<bool>,

    /// Conversation history for context
    pub history: Option<Vec<ChatMessage>>,
}

/// A chat message for conversation history
#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A source node used to generate the answer
#[derive(Serialize)]
pub struct SourceNode {
    pub node_id: String,
    pub content: String,
    pub similarity: f32,
    pub source: Option<String>,
}

/// Response from ask operation
#[derive(Serialize)]
pub struct AskResponse {
    pub success: bool,
    pub answer: Option<String>,
    pub sources: Vec<SourceNode>,
    pub latency_ms: u64,
    pub tokens_used: Option<u32>,
}

// ============================================================================
// Sync REM API
// ============================================================================

/// Request to trigger REM phase synchronization
#[derive(Deserialize, Default)]
pub struct SyncRemRequest {
    /// Only process specific namespace
    pub namespace: Option<String>,

    /// Maximum nodes to process
    pub max_nodes: Option<usize>,

    /// Whether to perform web search for incomplete nodes
    pub enable_web_search: Option<bool>,

    /// Whether to run RAPTOR clustering
    pub enable_clustering: Option<bool>,
}

/// Response from REM sync operation
#[derive(Serialize)]
pub struct SyncRemResponse {
    pub success: bool,
    pub nodes_processed: usize,
    pub nodes_created: usize,
    pub nodes_updated: usize,
    pub clusters_formed: usize,
    pub latency_ms: u64,
    pub message: String,
}

// ============================================================================
// Memory Update API
// ============================================================================

/// Request to update an existing memory node
#[derive(Deserialize)]
pub struct MemoryUpdateRequest {
    /// ID of the node to update
    pub node_id: String,

    /// New content (if updating)
    pub content: Option<String>,

    /// New status
    pub status: Option<String>,

    /// New tags
    pub tags: Option<Vec<String>>,

    /// Mark as deprecated
    pub deprecated: Option<bool>,
}

/// Response from memory update operation
#[derive(Serialize)]
pub struct MemoryUpdateResponse {
    pub success: bool,
    pub node_id: String,
    pub updated_fields: Vec<String>,
    pub message: String,
}

// ============================================================================
// Search API
// ============================================================================

/// Request for vector similarity search
#[derive(Deserialize)]
pub struct SearchRequest {
    /// Text to search for
    pub query: String,

    /// Namespace to search in
    pub namespace: Option<String>,

    /// Maximum results to return
    pub limit: Option<usize>,

    /// Minimum similarity threshold (0.0 to 1.0)
    pub threshold: Option<f32>,
}

/// Search result item
#[derive(Serialize)]
pub struct SearchResult {
    pub node_id: String,
    pub content: String,
    pub similarity: f32,
    pub source: Option<String>,
    pub created_at: String,
}

/// Response from search operation
#[derive(Serialize)]
pub struct SearchResponse {
    pub success: bool,
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub latency_ms: u64,
}

// ============================================================================
// Stats API
// ============================================================================

/// System statistics response
#[derive(Serialize)]
pub struct StatsResponse {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub namespaces: Vec<NamespaceStats>,
    pub cache_metrics: CacheStats,
    pub llm_metrics: LlmStats,
}

/// Statistics for a namespace
#[derive(Serialize)]
pub struct NamespaceStats {
    pub name: String,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Cache statistics
#[derive(Serialize)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f64,
    pub hits: u64,
    pub misses: u64,
}

/// LLM usage statistics
#[derive(Serialize)]
pub struct LlmStats {
    pub embedding_model: String,
    pub chat_model: String,
    pub summarizer_model: String,
    pub is_local: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_request_deserialization() {
        let json = r#"{
            "content": "Test content",
            "source": "test.txt",
            "source_type": "text",
            "namespace": "test",
            "tags": ["tag1", "tag2"]
        }"#;

        let request: IngestRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.content, "Test content");
        assert_eq!(request.source, Some("test.txt".to_string()));
        assert_eq!(request.tags, Some(vec!["tag1".to_string(), "tag2".to_string()]));
    }

    #[test]
    fn test_ask_request_minimal() {
        let json = r#"{"question": "What is Rust?"}"#;
        let request: AskRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.question, "What is Rust?");
        assert!(request.namespace.is_none());
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            service: "fractalmind".to_string(),
            version: "0.1.0".to_string(),
            components: HealthComponents {
                database: true,
                llm: true,
                cache: true,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("database"));
    }
}

// ============================================================================
// Model Management Types
// ============================================================================

/// Response for model upload
#[derive(Serialize)]
pub struct UploadModelResponse {
    pub success: bool,
    pub model_id: String,
    pub message: String,
}

/// Request to convert a model
#[derive(Deserialize)]
pub struct ConvertModelRequest {
    pub model_id: String,
}

/// Response from model conversion
#[derive(Serialize)]
pub struct ConvertModelResponse {
    pub success: bool,
    pub message: String,
}

/// Response listing all models
#[derive(Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}

/// Information about a fractal model
#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub status: String,
    pub architecture: Option<serde_json::Value>,
    pub file_size: u64,
    pub created_at: String,
}

/// Response with single model details
#[derive(Serialize)]
pub struct GetModelResponse {
    pub model: ModelInfo,
}

/// Response from model deletion
#[derive(Serialize)]
pub struct DeleteModelResponse {
    pub success: bool,
    pub message: String,
}

/// Request to update model strategy
#[derive(Deserialize)]
pub struct UpdateStrategyRequest {
    pub strategy: String, // "fractal" or "ollama"
    pub model_id: Option<String>, // Required if strategy is "fractal"
}

/// Response from strategy update
#[derive(Serialize)]
pub struct UpdateStrategyResponse {
    pub success: bool,
    pub current_strategy: String,
    pub message: String,
}

/// Information about an Ollama model
#[derive(Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: Option<OllamaModelDetails>,
}

/// Detailed information about an Ollama model
#[derive(Serialize)]
pub struct OllamaModelDetails {
    pub parent_model: Option<String>,
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

/// Response listing Ollama models
#[derive(Serialize)]
pub struct ListOllamaModelsResponse {
    pub models: Vec<OllamaModelInfo>,
}

// ============================================================================
// Chunked Upload Types
// ============================================================================

/// Request to initialize a chunked upload
#[derive(Deserialize)]
pub struct InitUploadRequest {
    pub filename: String,
    pub total_size: u64,
    pub chunk_size: u64,
}

/// Response from upload initialization
#[derive(Serialize)]
pub struct InitUploadResponse {
    pub upload_id: String,
    pub chunk_size: u64,
    pub total_chunks: u64,
}

/// Response from chunk upload
#[derive(Serialize)]
pub struct UploadChunkResponse {
    pub success: bool,
    pub chunk_index: u64,
    pub chunks_received: u64,
    pub total_chunks: u64,
}

/// Combined progress response
#[derive(Serialize)]
pub struct ProgressResponse {
    pub upload_progress: f32,      // 0-100
    pub conversion_progress: f32,  // 0-100
    pub status: String,            // "uploading", "finalizing", "converting", "ready", "failed"
    pub upload_speed_mbps: Option<f32>,
    pub chunks_received: Option<u64>,
    pub total_chunks: Option<u64>,
    pub current_phase: Option<String>,  // For conversion: "parsing", "clustering", etc.
}

/// Response from upload finalization
#[derive(Serialize)]
pub struct FinalizeUploadResponse {
    pub success: bool,
    pub model_id: String,
    pub message: String,
}

/// Response from upload cancellation
#[derive(Serialize)]
pub struct CancelUploadResponse {
    pub success: bool,
    pub message: String,
}
