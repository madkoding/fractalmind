//! API route definitions.

#![allow(dead_code)]

use axum::{
    extract::{DefaultBodyLimit, State, ws::WebSocketUpgrade},
    routing::{delete, get, patch, post},
    Router,
};

use super::handlers::{self, SharedState, SystemStatus};

/// Creates the API router with all routes configured
pub fn create_router(state: SharedState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(handlers::health_check))
        // WebSocket for real-time status
        .route("/ws/status", get(ws_status_handler))
        // API v1 routes
        .nest("/v1", api_v1_routes())
        // Stats
        .route("/stats", get(handlers::stats))
        // State
        .with_state(state)
}

async fn ws_status_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> axum::response::Response {
    let sender = {
        let state = state.read().await;
        state.status_sender.clone()
    };
    ws.on_upgrade(move |socket| ws_stream(socket, sender))
}

async fn ws_stream(
    mut socket: axum::extract::ws::WebSocket,
    sender: tokio::sync::broadcast::Sender<handlers::SystemStatus>,
) {
    let mut subscriber = sender.subscribe();

    // Send initial status immediately (don't wait for broadcast)
    let initial_status = SystemStatus {
        overall: "checking".to_string(),
        services: vec![
            handlers::ServiceStatus { name: "surrealdb".to_string(), healthy: false, message: None },
            handlers::ServiceStatus { name: "ollama".to_string(), healthy: false, message: None },
            handlers::ServiceStatus { name: "chat_provider".to_string(), healthy: false, message: None },
            handlers::ServiceStatus { name: "searxng".to_string(), healthy: false, message: None },
        ],
    };
    let msg = serde_json::to_string(&initial_status).unwrap_or_default();
    if socket.send(axum::extract::ws::Message::Text(msg)).await.is_err() {
        return;
    }

    let mut ping_count = 0usize;
    loop {
        tokio::select! {
            result = subscriber.recv() => {
                match result {
                    Ok(status) => {
                        let msg = serde_json::to_string(&status).unwrap_or_default();
                        if socket.send(axum::extract::ws::Message::Text(msg)).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(30)) => {
                if socket.send(axum::extract::ws::Message::Ping(vec![].into())).await.is_err() {
                    break;
                }
            }
        }
    }
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
        // LLM configuration (get and update)
        .route("/config/llm", patch(handlers::update_llm_config))
        .route("/config/llm", get(handlers::get_llm_config))
        // Model upload routes
        .nest("/models", model_upload_routes())
}

/// Model upload routes for chunked GGUF uploads
fn model_upload_routes() -> Router<SharedState> {
    Router::new()
        // Initialize chunked upload
        .route("/upload/init", post(handlers::init_model_upload))
        // Upload a chunk (increased body limit for 50MB+ chunks)
        .route(
            "/upload/:upload_id/chunk",
            post(handlers::upload_model_chunk),
        )
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // 100MB limit for chunks
        // Finalize upload
        .route(
            "/upload/:upload_id/finalize",
            post(handlers::finalize_model_upload),
        )
        // Get upload status
        .route(
            "/upload/:upload_id/status",
            get(handlers::get_upload_status),
        )
        // Cancel upload
        .route(
            "/upload/:upload_id/cancel",
            post(handlers::cancel_model_upload),
        )
        // Progress stream (SSE)
        .route(
            "/upload/:upload_id/progress",
            get(handlers::upload_progress_stream),
        )
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
