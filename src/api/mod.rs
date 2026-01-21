//! API module for Fractal-Mind.
//!
//! This module provides the HTTP REST API built with Axum:
//! - `/health` - Health check endpoint
//! - `/stats` - System statistics
//! - `/v1/ingest` - Ingest content into knowledge graph
//! - `/v1/remember` - Store episodic memory
//! - `/v1/ask` - Query knowledge graph with LLM response
//! - `/v1/search` - Vector similarity search
//! - `/v1/sync_rem` - Trigger REM phase synchronization
//! - `/v1/memory` - Update existing memory nodes

pub mod error;
pub mod handlers;
pub mod routes;
pub mod types;

// Re-exports
pub use error::{ApiError, ApiResult, ErrorResponse};
pub use handlers::{AppState, SharedState};
pub use routes::{create_router, print_routes};
pub use types::*;
