//! Upload session model for chunked file uploads.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Status of an upload session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum UploadStatus {
    Uploading,
    Finalizing,
    Converting,
    Ready,
    Failed,
    Cancelled,
}

impl UploadStatus {
    pub fn as_str(&self) -> &str {
        match self {
            UploadStatus::Uploading => "uploading",
            UploadStatus::Finalizing => "finalizing",
            UploadStatus::Converting => "converting",
            UploadStatus::Ready => "ready",
            UploadStatus::Failed => "failed",
            UploadStatus::Cancelled => "cancelled",
        }
    }
}

/// Represents an ongoing chunked upload session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadSession {
    /// Unique identifier for this upload session
    pub upload_id: String,
    
    /// Original filename
    pub filename: String,
    
    /// Total file size in bytes
    pub total_size: u64,
    
    /// Size of each chunk in bytes
    pub chunk_size: u64,
    
    /// Total number of chunks
    pub total_chunks: u64,
    
    /// Set of chunk indices that have been received
    pub chunks_received: Vec<u64>,
    
    /// Current status of the upload
    pub status: UploadStatus,
    
    /// Path to the temporary upload file
    pub temp_path: String,
    
    /// Model ID (set after finalization)
    pub model_id: Option<String>,
    
    /// Upload progress (0-100)
    pub upload_progress: f32,
    
    /// Conversion progress (0-100)
    pub conversion_progress: f32,
    
    /// Current conversion phase
    pub current_phase: Option<String>,
    
    /// Upload speed in MB/s (calculated from recent chunks)
    pub upload_speed_mbps: Option<f32>,
    
    /// Timestamp of last chunk received
    pub last_chunk_at: Option<DateTime<Utc>>,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl UploadSession {
    /// Create a new upload session
    pub fn new(filename: String, total_size: u64, chunk_size: u64) -> Self {
        let upload_id = format!("upload_{}", Uuid::new_v4().to_string().replace("-", ""));
        let total_chunks = (total_size + chunk_size - 1) / chunk_size;
        let temp_path = format!("/var/tmp/fractalmind_uploads/{}.part", upload_id);
        let now = Utc::now();

        Self {
            upload_id,
            filename,
            total_size,
            chunk_size,
            total_chunks,
            chunks_received: Vec::new(),
            status: UploadStatus::Uploading,
            temp_path,
            model_id: None,
            upload_progress: 0.0,
            conversion_progress: 0.0,
            current_phase: None,
            upload_speed_mbps: None,
            last_chunk_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Record that a chunk has been received
    pub fn add_chunk(&mut self, chunk_index: u64) {
        if !self.chunks_received.contains(&chunk_index) {
            self.chunks_received.push(chunk_index);
            self.chunks_received.sort();
        }
        
        self.upload_progress = (self.chunks_received.len() as f32 / self.total_chunks as f32) * 100.0;
        self.last_chunk_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Check if all chunks have been received
    pub fn is_complete(&self) -> bool {
        self.chunks_received.len() as u64 == self.total_chunks
    }

    /// Get missing chunk indices
    pub fn missing_chunks(&self) -> Vec<u64> {
        (0..self.total_chunks)
            .filter(|i| !self.chunks_received.contains(i))
            .collect()
    }

    /// Update conversion progress
    pub fn update_conversion_progress(&mut self, progress: f32, phase: Option<String>) {
        self.conversion_progress = progress.clamp(0.0, 100.0);
        self.current_phase = phase;
        self.updated_at = Utc::now();
    }

    /// Calculate upload speed based on recent activity
    pub fn calculate_speed(&mut self, chunk_size_bytes: u64, elapsed_secs: f32) {
        if elapsed_secs > 0.0 {
            let mb = chunk_size_bytes as f32 / (1024.0 * 1024.0);
            self.upload_speed_mbps = Some(mb / elapsed_secs);
        }
    }

    /// Mark upload as failed
    pub fn mark_failed(&mut self) {
        self.status = UploadStatus::Failed;
        self.updated_at = Utc::now();
    }

    /// Mark upload as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = UploadStatus::Cancelled;
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_upload_session() {
        let session = UploadSession::new("test.gguf".to_string(), 1000, 100);
        assert_eq!(session.filename, "test.gguf");
        assert_eq!(session.total_size, 1000);
        assert_eq!(session.chunk_size, 100);
        assert_eq!(session.total_chunks, 10);
        assert_eq!(session.chunks_received.len(), 0);
        assert_eq!(session.status, UploadStatus::Uploading);
    }

    #[test]
    fn test_add_chunk() {
        let mut session = UploadSession::new("test.gguf".to_string(), 1000, 100);
        session.add_chunk(0);
        session.add_chunk(1);
        
        assert_eq!(session.chunks_received.len(), 2);
        assert_eq!(session.upload_progress, 20.0);
    }

    #[test]
    fn test_is_complete() {
        let mut session = UploadSession::new("test.gguf".to_string(), 1000, 100);
        assert!(!session.is_complete());
        
        for i in 0..10 {
            session.add_chunk(i);
        }
        
        assert!(session.is_complete());
        assert_eq!(session.upload_progress, 100.0);
    }

    #[test]
    fn test_missing_chunks() {
        let mut session = UploadSession::new("test.gguf".to_string(), 1000, 100);
        session.add_chunk(0);
        session.add_chunk(2);
        session.add_chunk(4);
        
        let missing = session.missing_chunks();
        assert_eq!(missing, vec![1, 3, 5, 6, 7, 8, 9]);
    }
}
