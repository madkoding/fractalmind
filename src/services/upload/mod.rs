//! Upload session manager for chunked file uploads.
//!
//! Handles the lifecycle of upload sessions including:
//! - Session initialization
//! - Chunk reception and validation
//! - Upload finalization
//! - Session persistence and resumability
//! - Automatic cleanup of expired sessions

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::db::connection::DatabaseConnection;
use crate::models::upload_session::{UploadSession, UploadStatus};
use crate::services::storage::{StorageManager, DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, MAX_FILE_SIZE, MIN_CHUNK_SIZE};

/// Configuration for upload session manager
#[derive(Debug, Clone)]
pub struct UploadConfig {
    /// Default chunk size in bytes (100MB)
    pub default_chunk_size: u64,
    
    /// Minimum chunk size (10MB)
    pub min_chunk_size: u64,
    
    /// Maximum chunk size (500MB)
    pub max_chunk_size: u64,
    
    /// Maximum file size (500GB)
    pub max_file_size: u64,
    
    /// Session expiration time in hours
    pub session_expiry_hours: i64,
    
    /// Cleanup interval in minutes
    pub cleanup_interval_minutes: u64,
}

impl Default for UploadConfig {
    fn default() -> Self {
        Self {
            default_chunk_size: DEFAULT_CHUNK_SIZE,
            min_chunk_size: MIN_CHUNK_SIZE,
            max_chunk_size: MAX_CHUNK_SIZE,
            max_file_size: MAX_FILE_SIZE,
            session_expiry_hours: 24,
            cleanup_interval_minutes: 60,
        }
    }
}

/// Result of a chunk upload operation
#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub success: bool,
    pub chunk_index: u64,
    pub chunks_received: u64,
    pub total_chunks: u64,
    pub checksum: String,
    pub upload_speed_mbps: Option<f32>,
}

/// Result of upload finalization
#[derive(Debug, Clone)]
pub struct FinalizeResult {
    pub success: bool,
    pub model_id: String,
    pub file_path: String,
    pub file_size: u64,
}

/// Manages upload sessions and coordinates with storage
pub struct UploadSessionManager {
    /// In-memory session cache
    sessions: Arc<RwLock<HashMap<String, UploadSession>>>,
    
    /// Storage manager for file operations
    storage: StorageManager,
    
    /// Database connection for persistence
    db: Option<Arc<DatabaseConnection>>,
    
    /// Configuration
    config: UploadConfig,
}

impl UploadSessionManager {
    /// Create a new upload session manager
    pub fn new(storage: StorageManager) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            storage,
            db: None,
            config: UploadConfig::default(),
        }
    }
    
    /// Create with database connection for persistence
    pub fn with_db(storage: StorageManager, db: Arc<DatabaseConnection>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            storage,
            db: Some(db),
            config: UploadConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(mut self, config: UploadConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Initialize the manager (creates directories, loads persisted sessions)
    pub async fn init(&self) -> Result<()> {
        // Initialize storage
        self.storage.init().await?;
        
        // Load persisted sessions from database if available
        if let Some(db) = &self.db {
            self.load_sessions_from_db(db).await?;
        }
        
        info!("Upload session manager initialized");
        Ok(())
    }
    
    /// Initialize a new upload session
    pub async fn init_upload(
        &self,
        filename: String,
        total_size: u64,
        chunk_size: Option<u64>,
    ) -> Result<UploadSession> {
        // Validate filename
        if !filename.to_lowercase().ends_with(".gguf") {
            return Err(anyhow::anyhow!("File must be a .gguf model file"));
        }
        
        // Validate file size
        if total_size == 0 {
            return Err(anyhow::anyhow!("File size cannot be 0"));
        }
        
        if total_size > self.config.max_file_size {
            return Err(anyhow::anyhow!(
                "File size {} exceeds maximum allowed {} bytes ({}GB)",
                total_size,
                self.config.max_file_size,
                self.config.max_file_size / (1024 * 1024 * 1024)
            ));
        }
        
        // Determine chunk size
        let chunk_size = chunk_size.unwrap_or(self.config.default_chunk_size);
        let chunk_size = chunk_size.clamp(self.config.min_chunk_size, self.config.max_chunk_size);
        
        // Create session
        let session = UploadSession::new(filename, total_size, chunk_size);
        let upload_id = session.upload_id.clone();
        
        // Create temp file
        self.storage.create_temp_file(&upload_id).await?;
        
        // Preallocate file space
        self.storage.preallocate(&upload_id, total_size).await?;
        
        // Store session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.clone(), session.clone());
        }
        
        // Persist to database
        if let Some(db) = &self.db {
            self.save_session_to_db(db, &session).await?;
        }
        
        info!(
            "Initialized upload session: id={}, file={}, size={}, chunks={}",
            upload_id,
            session.filename,
            session.total_size,
            session.total_chunks
        );
        
        Ok(session)
    }
    
    /// Upload a chunk
    pub async fn upload_chunk(
        &self,
        upload_id: &str,
        chunk_index: u64,
        data: bytes::Bytes,
        checksum: Option<&str>,
    ) -> Result<ChunkResult> {
        let start = Instant::now();
        
        // Get session
        let mut session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(upload_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Upload session not found: {}", upload_id))?
        };
        
        // Validate session state
        if session.status != UploadStatus::Uploading {
            return Err(anyhow::anyhow!(
                "Upload session is not in uploading state: {:?}",
                session.status
            ));
        }
        
        // Validate chunk index
        if chunk_index >= session.total_chunks {
            return Err(anyhow::anyhow!(
                "Invalid chunk index {} (total chunks: {})",
                chunk_index,
                session.total_chunks
            ));
        }
        
        // Check if chunk already received
        if session.chunks_received.contains(&chunk_index) {
            warn!("Chunk {} already received for upload {}", chunk_index, upload_id);
            // Return success anyway (idempotent)
            return Ok(ChunkResult {
                success: true,
                chunk_index,
                chunks_received: session.chunks_received.len() as u64,
                total_chunks: session.total_chunks,
                checksum: "already_received".to_string(),
                upload_speed_mbps: session.upload_speed_mbps,
            });
        }
        
        // Write chunk to disk with checksum verification
        let written_checksum = self
            .storage
            .append_chunk(upload_id, chunk_index, session.chunk_size, &data, checksum)
            .await?;
        
        // Update session
        let elapsed = start.elapsed().as_secs_f32();
        session.add_chunk(chunk_index);
        session.calculate_speed(data.len() as u64, elapsed);
        
        let result = ChunkResult {
            success: true,
            chunk_index,
            chunks_received: session.chunks_received.len() as u64,
            total_chunks: session.total_chunks,
            checksum: written_checksum,
            upload_speed_mbps: session.upload_speed_mbps,
        };
        
        // Store updated session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.to_string(), session.clone());
        }
        
        // Persist to database
        if let Some(db) = &self.db {
            self.save_session_to_db(db, &session).await?;
        }
        
        debug!(
            "Received chunk {} for upload {} ({}/{}), speed: {:?} MB/s",
            chunk_index,
            upload_id,
            result.chunks_received,
            result.total_chunks,
            result.upload_speed_mbps
        );
        
        Ok(result)
    }
    
    /// Finalize an upload
    pub async fn finalize(&self, upload_id: &str) -> Result<FinalizeResult> {
        // Get session
        let mut session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(upload_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Upload session not found: {}", upload_id))?
        };
        
        // Check all chunks received
        if !session.is_complete() {
            let missing = session.missing_chunks();
            return Err(anyhow::anyhow!(
                "Upload incomplete: missing {} chunks: {:?}",
                missing.len(),
                if missing.len() > 10 { &missing[..10] } else { &missing }
            ));
        }
        
        // Update status
        session.status = UploadStatus::Finalizing;
        session.updated_at = Utc::now();
        
        // Verify file size
        let file_verified = self
            .storage
            .verify_file(
                &std::path::PathBuf::from(&session.temp_path),
                session.total_size,
            )
            .await?;
        
        if !file_verified {
            session.status = UploadStatus::Failed;
            return Err(anyhow::anyhow!("File verification failed"));
        }
        
        // Generate unique filename to avoid collisions
        let model_id = format!("model_{}", uuid::Uuid::new_v4().to_string().replace("-", ""));
        let final_filename = format!("{}_{}", model_id, session.filename);
        
        // Move to final location
        let final_path = self.storage.finalize(upload_id, &final_filename).await?;
        
        // Update session
        session.status = UploadStatus::Converting;
        session.model_id = Some(model_id.clone());
        session.updated_at = Utc::now();
        
        // Store updated session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.to_string(), session.clone());
        }
        
        // Persist to database
        if let Some(db) = &self.db {
            self.save_session_to_db(db, &session).await?;
        }
        
        info!(
            "Finalized upload {} -> model_id={}, path={:?}",
            upload_id, model_id, final_path
        );
        
        Ok(FinalizeResult {
            success: true,
            model_id,
            file_path: final_path.to_string_lossy().to_string(),
            file_size: session.total_size,
        })
    }
    
    /// Get upload status
    pub async fn get_status(&self, upload_id: &str) -> Option<UploadSession> {
        let sessions = self.sessions.read().await;
        sessions.get(upload_id).cloned()
    }
    
    /// Cancel an upload
    pub async fn cancel(&self, upload_id: &str) -> Result<()> {
        // Get and update session
        let mut session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(upload_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Upload session not found: {}", upload_id))?
        };
        
        session.mark_cancelled();
        
        // Cleanup temp files
        self.storage.cleanup(upload_id).await?;
        
        // Store updated session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.to_string(), session.clone());
        }
        
        // Persist to database
        if let Some(db) = &self.db {
            self.save_session_to_db(db, &session).await?;
        }
        
        info!("Cancelled upload {}", upload_id);
        Ok(())
    }
    
    /// Get expired sessions (older than configured expiry time)
    pub async fn get_expired_sessions(&self) -> Vec<String> {
        let expiry_threshold = Utc::now() - Duration::hours(self.config.session_expiry_hours);
        
        let sessions = self.sessions.read().await;
        sessions
            .iter()
            .filter(|(_, session)| {
                session.status == UploadStatus::Uploading
                    && session.created_at < expiry_threshold
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Cleanup expired sessions
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let expired = self.get_expired_sessions().await;
        let count = expired.len();
        
        for upload_id in expired {
            if let Err(e) = self.cancel(&upload_id).await {
                error!("Failed to cleanup expired upload {}: {}", upload_id, e);
            }
        }
        
        if count > 0 {
            info!("Cleaned up {} expired upload sessions", count);
        }
        
        Ok(count)
    }
    
    /// Update conversion progress for a session
    pub async fn update_conversion_progress(
        &self,
        upload_id: &str,
        progress: f32,
        phase: Option<String>,
    ) -> Result<()> {
        let mut session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(upload_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Upload session not found: {}", upload_id))?
        };
        
        session.update_conversion_progress(progress, phase);
        
        // Store updated session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.to_string(), session.clone());
        }
        
        Ok(())
    }
    
    /// Mark upload as ready
    pub async fn mark_ready(&self, upload_id: &str) -> Result<()> {
        let mut session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(upload_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Upload session not found: {}", upload_id))?
        };
        
        session.status = UploadStatus::Ready;
        session.conversion_progress = 100.0;
        session.updated_at = Utc::now();
        
        // Store updated session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.to_string(), session.clone());
        }
        
        // Persist to database
        if let Some(db) = &self.db {
            self.save_session_to_db(db, &session).await?;
        }
        
        info!("Upload {} marked as ready", upload_id);
        Ok(())
    }
    
    /// Mark upload as failed
    pub async fn mark_failed(&self, upload_id: &str, _error: &str) -> Result<()> {
        let mut session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(upload_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Upload session not found: {}", upload_id))?
        };
        
        session.mark_failed();
        
        // Store updated session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(upload_id.to_string(), session.clone());
        }
        
        // Persist to database
        if let Some(db) = &self.db {
            self.save_session_to_db(db, &session).await?;
        }
        
        error!("Upload {} marked as failed", upload_id);
        Ok(())
    }
    
    // ========================================================================
    // Database persistence helpers
    // ========================================================================
    
    async fn save_session_to_db(&self, db: &DatabaseConnection, session: &UploadSession) -> Result<()> {
        let query = r#"
            UPSERT upload_sessions SET
                upload_id = $upload_id,
                filename = $filename,
                total_size = $total_size,
                chunk_size = $chunk_size,
                total_chunks = $total_chunks,
                chunks_received = $chunks_received,
                status = $status,
                temp_path = $temp_path,
                model_id = $model_id,
                upload_progress = $upload_progress,
                conversion_progress = $conversion_progress,
                current_phase = $current_phase,
                created_at = $created_at,
                updated_at = $updated_at
            WHERE upload_id = $upload_id
        "#;
        
        let chunks_json = serde_json::to_string(&session.chunks_received)?;
        
        db.query(query)
            .bind(("upload_id", &session.upload_id))
            .bind(("filename", &session.filename))
            .bind(("total_size", session.total_size as i64))
            .bind(("chunk_size", session.chunk_size as i64))
            .bind(("total_chunks", session.total_chunks as i64))
            .bind(("chunks_received", chunks_json))
            .bind(("status", session.status.as_str()))
            .bind(("temp_path", &session.temp_path))
            .bind(("model_id", &session.model_id))
            .bind(("upload_progress", session.upload_progress))
            .bind(("conversion_progress", session.conversion_progress))
            .bind(("current_phase", &session.current_phase))
            .bind(("created_at", session.created_at))
            .bind(("updated_at", session.updated_at))
            .await
            .context("Failed to save upload session to database")?;
        
        Ok(())
    }
    
    async fn load_sessions_from_db(&self, db: &DatabaseConnection) -> Result<()> {
        let query = r#"
            SELECT * FROM upload_sessions 
            WHERE status IN ["uploading", "finalizing", "converting"]
        "#;
        
        let result = db.query(query).await;
        
        match result {
            Ok(mut response) => {
                let sessions: Vec<serde_json::Value> = response.take(0)?;
                
                let mut cache = self.sessions.write().await;
                for session_json in sessions {
                    if let Ok(session) = serde_json::from_value::<UploadSession>(session_json) {
                        cache.insert(session.upload_id.clone(), session);
                    }
                }
                
                info!("Loaded {} active upload sessions from database", cache.len());
            }
            Err(e) => {
                warn!("Could not load sessions from database (table may not exist yet): {}", e);
            }
        }
        
        Ok(())
    }
}

/// Cleanup job that runs periodically
pub struct UploadCleanupJob {
    manager: Arc<UploadSessionManager>,
    interval_minutes: u64,
}

impl UploadCleanupJob {
    pub fn new(manager: Arc<UploadSessionManager>, interval_minutes: u64) -> Self {
        Self {
            manager,
            interval_minutes,
        }
    }
    
    /// Start the cleanup job (runs in background)
    pub fn start(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let interval = tokio::time::Duration::from_secs(self.interval_minutes * 60);
            let mut ticker = tokio::time::interval(interval);
            
            loop {
                ticker.tick().await;
                
                match self.manager.cleanup_expired().await {
                    Ok(count) => {
                        if count > 0 {
                            debug!("Cleanup job removed {} expired sessions", count);
                        }
                    }
                    Err(e) => {
                        error!("Cleanup job error: {}", e);
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn test_manager() -> (UploadSessionManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = StorageManager::with_paths(
            temp_dir.path().join("uploads"),
            temp_dir.path().join("models"),
        );
        
        let manager = UploadSessionManager::new(storage);
        manager.init().await.unwrap();
        
        (manager, temp_dir)
    }
    
    #[tokio::test]
    async fn test_init_upload() {
        let (manager, _temp) = test_manager().await;
        
        let session = manager
            .init_upload("test.gguf".to_string(), 1_000_000, None)
            .await
            .unwrap();
        
        assert!(session.upload_id.starts_with("upload_"));
        assert_eq!(session.filename, "test.gguf");
        assert_eq!(session.total_size, 1_000_000);
        assert_eq!(session.status, UploadStatus::Uploading);
    }
    
    #[tokio::test]
    async fn test_invalid_file_extension() {
        let (manager, _temp) = test_manager().await;
        
        let result = manager
            .init_upload("test.txt".to_string(), 1_000_000, None)
            .await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(".gguf"));
    }
    
    #[tokio::test]
    async fn test_upload_chunk() {
        let (manager, _temp) = test_manager().await;
        
        let session = manager
            .init_upload("test.gguf".to_string(), 1000, Some(100))
            .await
            .unwrap();
        
        let data = bytes::Bytes::from(vec![0u8; 100]);
        let result = manager
            .upload_chunk(&session.upload_id, 0, data, None)
            .await
            .unwrap();
        
        assert!(result.success);
        assert_eq!(result.chunk_index, 0);
        assert_eq!(result.chunks_received, 1);
    }
    
    #[tokio::test]
    async fn test_complete_upload_flow() {
        let (manager, _temp) = test_manager().await;
        
        let chunk_size = 100u64;
        let total_size = 300u64;
        
        let session = manager
            .init_upload("model.gguf".to_string(), total_size, Some(chunk_size))
            .await
            .unwrap();
        
        // Upload all chunks
        for i in 0..3 {
            let data = bytes::Bytes::from(vec![i as u8; 100]);
            manager
                .upload_chunk(&session.upload_id, i, data, None)
                .await
                .unwrap();
        }
        
        // Finalize
        let result = manager.finalize(&session.upload_id).await.unwrap();
        
        assert!(result.success);
        assert!(result.model_id.starts_with("model_"));
    }
    
    #[tokio::test]
    async fn test_cancel_upload() {
        let (manager, _temp) = test_manager().await;
        
        let session = manager
            .init_upload("test.gguf".to_string(), 1000, None)
            .await
            .unwrap();
        
        manager.cancel(&session.upload_id).await.unwrap();
        
        let status = manager.get_status(&session.upload_id).await.unwrap();
        assert_eq!(status.status, UploadStatus::Cancelled);
    }
}
