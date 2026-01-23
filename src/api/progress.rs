//! Progress tracking for long-running ingestion operations.

use axum::{
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Extension,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;
use uuid::Uuid;

/// Progress state for a file ingestion operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionProgress {
    /// Unique session ID for this ingestion
    pub session_id: String,
    
    /// Total number of chunks to process
    pub total_chunks: usize,
    
    /// Current chunk being embedded
    pub current_chunk: usize,
    
    /// Number of embeddings completed
    pub embeddings_completed: usize,
    
    /// Number of nodes persisted to DB
    pub nodes_persisted: usize,
    
    /// Current stage: "extracting", "chunking", "embedding", "persisting", "complete"
    pub stage: String,
    
    /// Optional progress message
    pub message: Option<String>,
    
    /// Whether the operation completed successfully
    pub success: bool,
    
    /// Any error message if failed
    pub error: Option<String>,
}

impl IngestionProgress {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            total_chunks: 0,
            current_chunk: 0,
            embeddings_completed: 0,
            nodes_persisted: 0,
            stage: "initializing".to_string(),
            message: None,
            success: false,
            error: None,
        }
    }

    pub fn with_stage(mut self, stage: &str, message: Option<String>) -> Self {
        self.stage = stage.to_string();
        self.message = message;
        self
    }

    pub fn with_total_chunks(mut self, total: usize) -> Self {
        self.total_chunks = total;
        self
    }

    pub fn embedding_progress(&self) -> f32 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        (self.embeddings_completed as f32 / self.total_chunks as f32) * 100.0
    }

    pub fn persistence_progress(&self) -> f32 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        (self.nodes_persisted as f32 / self.total_chunks as f32) * 100.0
    }

    pub fn overall_progress(&self) -> f32 {
        match self.stage.as_str() {
            "extracting" => 10.0,
            "chunking" => 20.0,
            "embedding" => 20.0 + (self.embedding_progress() * 0.5),
            "persisting" => 70.0 + (self.persistence_progress() * 0.25),
            "complete" => 100.0,
            _ => 0.0,
        }
    }
}

/// Global progress tracker shared across handlers.
pub type ProgressTracker = Arc<RwLock<HashMap<String, IngestionProgress>>>;

/// Creates a new progress tracker.
pub fn create_progress_tracker() -> ProgressTracker {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Registers a new ingestion session and returns the session ID.
pub async fn register_session(tracker: &ProgressTracker) -> String {
    let session_id = Uuid::new_v4().to_string();
    let progress = IngestionProgress::new(session_id.clone());
    
    let mut map = tracker.write().await;
    map.insert(session_id.clone(), progress);
    
    debug!("Registered new ingestion session: {}", session_id);
    session_id
}

/// Updates progress for a session.
pub async fn update_progress(
    tracker: &ProgressTracker,
    session_id: &str,
    update_fn: impl FnOnce(&mut IngestionProgress),
) {
    let mut map = tracker.write().await;
    if let Some(progress) = map.get_mut(session_id) {
        update_fn(progress);
    }
}

/// Gets current progress for a session.
pub async fn get_progress(tracker: &ProgressTracker, session_id: &str) -> Option<IngestionProgress> {
    let map = tracker.read().await;
    map.get(session_id).cloned()
}

/// Removes a completed session after a delay (for cleanup).
pub async fn cleanup_session(tracker: &ProgressTracker, session_id: String, delay_secs: u64) {
    tokio::time::sleep(tokio::time::Duration::from_secs(delay_secs)).await;
    
    let mut map = tracker.write().await;
    map.remove(&session_id);
    debug!("Cleaned up ingestion session: {}", session_id);
}

/// SSE stream handler for real-time progress updates.
pub async fn progress_stream(
    Extension(tracker): Extension<ProgressTracker>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(250));
        
        loop {
            interval.tick().await;
            
            // Get current progress
            let progress = {
                let map = tracker.read().await;
                map.get(&session_id).cloned()
            };
            
            match progress {
                Some(p) => {
                    // Serialize progress to JSON
                    let json = match serde_json::to_string(&p) {
                        Ok(j) => j,
                        Err(e) => {
                            debug!("Failed to serialize progress: {}", e);
                            continue;
                        }
                    };
                    
                    // Send SSE event
                    yield Ok::<_, Infallible>(Event::default().data(json));
                    
                    // Stop streaming when complete or failed
                    if p.stage == "complete" || p.error.is_some() {
                        break;
                    }
                }
                None => {
                    // Session not found - might have been cleaned up
                    let error_msg = serde_json::json!({
                        "error": "Session not found",
                        "session_id": session_id
                    });
                    yield Ok::<_, Infallible>(
                        Event::default().event("error").data(error_msg.to_string())
                    );
                    break;
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
