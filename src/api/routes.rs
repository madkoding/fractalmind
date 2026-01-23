//! API route definitions.

#![allow(dead_code)]

use axum::{
    extract::DefaultBodyLimit,
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
        // Build fractal structure
        .route("/build-fractal", post(handlers::build_fractal))
        // REM phase sync
        .route("/sync_rem", post(handlers::sync_rem))
        // Memory management
        .route("/memory", patch(handlers::memory_update))
        // Model upload routes
        .nest("/models", model_upload_routes())
}

/// Model upload routes for chunked GGUF uploads
fn model_upload_routes() -> Router<SharedState> {
    Router::new()
        // Initialize chunked upload
        .route("/upload/init", post(handlers::init_model_upload))
        // Upload a chunk (increased body limit for 50MB+ chunks)
        .route("/upload/:upload_id/chunk", post(handlers::upload_model_chunk))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // 100MB limit for chunks
        // Finalize upload
        .route("/upload/:upload_id/finalize", post(handlers::finalize_model_upload))
        // Get upload status
        .route("/upload/:upload_id/status", get(handlers::get_upload_status))
        // Cancel upload
        .route("/upload/:upload_id/cancel", post(handlers::cancel_model_upload))
        // Progress stream (SSE)
        .route("/upload/:upload_id/progress", get(handlers::upload_progress_stream))
        // List Ollama models
        .route("/ollama", get(handlers::list_ollama_models))
        // List all fractal models
        .route("/", get(handlers::list_models))
        // Get specific model
        .route("/:model_id", get(handlers::get_model))
        // Delete model
        .route("/:model_id", delete(handlers::delete_model))
        // Start conversion
        .route("/:model_id/convert", post(handlers::convert_model))
}

/// Prints all available routes for logging
pub fn print_routes() {
    tracing::info!("Available API routes:");
    tracing::info!("  GET  /health         - Health check with component status");
    tracing::info!("  GET  /stats          - System statistics");
    tracing::info!("  POST /v1/ingest      - Ingest content into knowledge graph");
    tracing::info!("  POST /v1/ingest/file - Ingest file (multipart)");
    tracing::info!("  POST /v1/remember    - Store episodic memory");
    tracing::info!("  POST /v1/ask         - Query knowledge graph with LLM");
    tracing::info!("  POST /v1/search      - Vector similarity search + SSSP navigation");
    tracing::info!("  POST /v1/build-fractal - Build fractal hierarchy (RAPTOR)");
    tracing::info!("  POST /v1/sync_rem    - Trigger REM phase synchronization");
    tracing::info!("  PATCH /v1/memory     - Update existing memory node");
    tracing::info!("Model upload routes:");
    tracing::info!("  POST /v1/models/upload/init         - Initialize chunked upload");
    tracing::info!("  POST /v1/models/upload/:id/chunk    - Upload a chunk");
    tracing::info!("  POST /v1/models/upload/:id/finalize - Finalize upload");
    tracing::info!("  GET  /v1/models/upload/:id/status   - Get upload status");
    tracing::info!("  POST /v1/models/upload/:id/cancel   - Cancel upload");
    tracing::info!("  GET  /v1/models/upload/:id/progress - SSE progress stream");
    tracing::info!("  GET  /v1/models                     - List all models");
    tracing::info!("  GET  /v1/models/:id                 - Get model details");
    tracing::info!("  DELETE /v1/models/:id               - Delete model");
    tracing::info!("  POST /v1/models/:id/convert         - Start model conversion");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_print_routes_no_panic() {
        // Just verify it doesn't panic
        // Note: actual logging won't happen in tests without setup
    }
}
