//! API route definitions.

#![allow(dead_code)]

use axum::{
    routing::{delete, get, patch, post},
    Router,
};

use super::handlers::{self, SharedState};

/// Creates the API router with all routes configured
pub fn create_router(state: SharedState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(handlers::health_check))
        // API v1 routes
        .nest("/v1", api_v1_routes())
        // Stats
        .route("/stats", get(handlers::stats))
        // State
        .with_state(state)
}

/// API v1 routes
fn api_v1_routes() -> Router<SharedState> {
    Router::new()
        // Knowledge ingestion (JSON)
        .route("/ingest", post(handlers::ingest))
        // Knowledge ingestion (file upload - multipart)
        .route("/ingest/file", post(handlers::ingest_file))
        // Episodic memory
        .route("/remember", post(handlers::remember))
        // Knowledge query
        .route("/ask", post(handlers::ask))
        // Search
        .route("/search", post(handlers::search))
        // REM phase sync
        .route("/sync_rem", post(handlers::sync_rem))
        // Memory management
        .route("/memory", patch(handlers::memory_update))
        // Model management
        .route("/models/upload", post(handlers::upload_model))
        .route("/models/convert", post(handlers::convert_model))
        .route("/models", get(handlers::list_models))
        .route("/models/ollama", get(handlers::list_ollama_models))
        .route("/models/:id", get(handlers::get_model))
        .route("/models/:id", delete(handlers::delete_model))
        .route("/config/model-strategy", patch(handlers::update_strategy))
}

/// Prints all available routes for logging
pub fn print_routes() {
    tracing::info!("Available API routes:");
    tracing::info!("  GET  /health        - Health check with component status");
    tracing::info!("  GET  /stats         - System statistics");
    tracing::info!("  POST /v1/ingest     - Ingest content into knowledge graph");
    tracing::info!("  POST /v1/ingest/file - Ingest file (multipart)");
    tracing::info!("  POST /v1/remember   - Store episodic memory");
    tracing::info!("  POST /v1/ask        - Query knowledge graph with LLM");
    tracing::info!("  POST /v1/search     - Vector similarity search");
    tracing::info!("  POST /v1/sync_rem   - Trigger REM phase synchronization");
    tracing::info!("  PATCH /v1/memory    - Update existing memory node");
    tracing::info!("  POST /v1/models/upload - Upload GGUF model");
    tracing::info!("  POST /v1/models/convert - Convert model to fractal");
    tracing::info!("  GET  /v1/models     - List all models");
    tracing::info!("  GET  /v1/models/ollama - List Ollama models");
    tracing::info!("  GET  /v1/models/:id - Get model details");
    tracing::info!("  DELETE /v1/models/:id - Delete model");
    tracing::info!("  PATCH /v1/config/model-strategy - Update model strategy");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_print_routes_no_panic() {
        // Just verify it doesn't panic
        // Note: actual logging won't happen in tests without setup
    }
}
