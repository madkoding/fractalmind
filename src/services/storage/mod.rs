//! Storage manager for handling large file uploads.
//!
//! Provides disk-based storage for chunked uploads without loading
//! entire files into memory. Supports files up to 500GB.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tracing::{debug, info, warn};

/// Default base path for temporary uploads
pub const DEFAULT_UPLOAD_PATH: &str = "/var/tmp/fractalmind_uploads";

/// Default chunk size: 100MB
pub const DEFAULT_CHUNK_SIZE: u64 = 100 * 1024 * 1024;

/// Maximum allowed file size: 500GB
pub const MAX_FILE_SIZE: u64 = 500 * 1024 * 1024 * 1024;

/// Minimum chunk size: 10MB
pub const MIN_CHUNK_SIZE: u64 = 10 * 1024 * 1024;

/// Maximum chunk size: 500MB
pub const MAX_CHUNK_SIZE: u64 = 500 * 1024 * 1024;

/// Storage manager for handling large file uploads
#[derive(Debug, Clone)]
pub struct StorageManager {
    /// Base path for storing temporary upload files
    base_path: PathBuf,
    
    /// Maximum chunk size in bytes
    max_chunk_size: u64,
    
    /// Final destination path for completed uploads
    models_path: PathBuf,
}

impl StorageManager {
    /// Create a new storage manager with default paths
    pub fn new() -> Self {
        Self {
            base_path: PathBuf::from(DEFAULT_UPLOAD_PATH),
            max_chunk_size: DEFAULT_CHUNK_SIZE,
            models_path: PathBuf::from("/var/tmp/fractalmind_models"),
        }
    }
    
    /// Create a storage manager with custom paths
    pub fn with_paths(base_path: PathBuf, models_path: PathBuf) -> Self {
        Self {
            base_path,
            max_chunk_size: DEFAULT_CHUNK_SIZE,
            models_path,
        }
    }
    
    /// Set maximum chunk size
    pub fn with_chunk_size(mut self, chunk_size: u64) -> Self {
        self.max_chunk_size = chunk_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        self
    }
    
    /// Initialize storage directories
    pub async fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.base_path)
            .await
            .context("Failed to create upload directory")?;
        
        fs::create_dir_all(&self.models_path)
            .await
            .context("Failed to create models directory")?;
        
        info!("Storage manager initialized: uploads={:?}, models={:?}", 
              self.base_path, self.models_path);
        Ok(())
    }
    
    /// Get the base path for uploads
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
    
    /// Get the models path
    pub fn models_path(&self) -> &Path {
        &self.models_path
    }
    
    /// Create a new temporary file for an upload
    pub async fn create_temp_file(&self, upload_id: &str) -> Result<PathBuf> {
        let temp_path = self.base_path.join(format!("{}.part", upload_id));
        
        // Create the file (truncate if exists)
        File::create(&temp_path)
            .await
            .with_context(|| format!("Failed to create temp file: {:?}", temp_path))?;
        
        debug!("Created temp file: {:?}", temp_path);
        Ok(temp_path)
    }
    
    /// Pre-allocate file space for the expected total size
    pub async fn preallocate(&self, upload_id: &str, total_size: u64) -> Result<()> {
        let temp_path = self.base_path.join(format!("{}.part", upload_id));
        
        let file = OpenOptions::new()
            .write(true)
            .open(&temp_path)
            .await
            .with_context(|| format!("Failed to open temp file for preallocation: {:?}", temp_path))?;
        
        // Set file length (sparse file on most filesystems)
        file.set_len(total_size)
            .await
            .with_context(|| format!("Failed to preallocate {} bytes", total_size))?;
        
        debug!("Preallocated {} bytes for upload {}", total_size, upload_id);
        Ok(())
    }
    
    /// Append a chunk to the upload file at the specified offset
    /// 
    /// # Arguments
    /// * `upload_id` - The upload session ID
    /// * `chunk_index` - The index of this chunk (0-based)
    /// * `chunk_size` - Expected size of each chunk
    /// * `data` - The chunk data
    /// * `expected_checksum` - Optional SHA256 checksum to verify
    /// 
    /// Returns the actual SHA256 checksum of the written data
    pub async fn append_chunk(
        &self,
        upload_id: &str,
        chunk_index: u64,
        chunk_size: u64,
        data: &[u8],
        expected_checksum: Option<&str>,
    ) -> Result<String> {
        let temp_path = self.base_path.join(format!("{}.part", upload_id));
        
        // Calculate checksum
        let mut hasher = Sha256::new();
        hasher.update(data);
        let checksum = format!("{:x}", hasher.finalize());
        
        // Verify checksum if provided
        if let Some(expected) = expected_checksum {
            if checksum != expected {
                return Err(anyhow::anyhow!(
                    "Checksum mismatch: expected {}, got {}",
                    expected,
                    checksum
                ));
            }
            debug!("Checksum verified for chunk {}", chunk_index);
        }
        
        // Calculate offset
        let offset = chunk_index * chunk_size;
        
        // Open file and seek to position
        let mut file = OpenOptions::new()
            .write(true)
            .open(&temp_path)
            .await
            .with_context(|| format!("Failed to open temp file: {:?}", temp_path))?;
        
        file.seek(SeekFrom::Start(offset))
            .await
            .with_context(|| format!("Failed to seek to offset {}", offset))?;
        
        // Write data
        file.write_all(data)
            .await
            .with_context(|| format!("Failed to write chunk {} at offset {}", chunk_index, offset))?;
        
        file.flush().await?;
        
        debug!(
            "Wrote chunk {} ({} bytes) at offset {} for upload {}",
            chunk_index,
            data.len(),
            offset,
            upload_id
        );
        
        Ok(checksum)
    }
    
    /// Finalize an upload by moving it to the models directory
    /// 
    /// Returns the final path of the model file
    pub async fn finalize(&self, upload_id: &str, filename: &str) -> Result<PathBuf> {
        let temp_path = self.base_path.join(format!("{}.part", upload_id));
        let final_path = self.models_path.join(filename);
        
        // Ensure temp file exists
        if !temp_path.exists() {
            return Err(anyhow::anyhow!(
                "Temp file not found: {:?}",
                temp_path
            ));
        }
        
        // Move file to final destination
        fs::rename(&temp_path, &final_path)
            .await
            .with_context(|| format!(
                "Failed to move {:?} to {:?}",
                temp_path, final_path
            ))?;
        
        info!("Finalized upload {} to {:?}", upload_id, final_path);
        Ok(final_path)
    }
    
    /// Verify the integrity of a completed upload
    pub async fn verify_file(&self, path: &Path, expected_size: u64) -> Result<bool> {
        let metadata = fs::metadata(path)
            .await
            .with_context(|| format!("Failed to get metadata for {:?}", path))?;
        
        if metadata.len() != expected_size {
            warn!(
                "File size mismatch: expected {}, got {}",
                expected_size,
                metadata.len()
            );
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Calculate SHA256 checksum of a file (streaming, memory-efficient)
    pub async fn calculate_file_checksum(&self, path: &Path) -> Result<String> {
        let mut file = File::open(path)
            .await
            .with_context(|| format!("Failed to open file for checksum: {:?}", path))?;
        
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
        
        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Clean up temporary files for an upload
    pub async fn cleanup(&self, upload_id: &str) -> Result<()> {
        let temp_path = self.base_path.join(format!("{}.part", upload_id));
        
        if temp_path.exists() {
            fs::remove_file(&temp_path)
                .await
                .with_context(|| format!("Failed to remove temp file: {:?}", temp_path))?;
            
            debug!("Cleaned up temp file for upload {}", upload_id);
        }
        
        Ok(())
    }
    
    /// Delete a finalized model file
    pub async fn delete_model(&self, filename: &str) -> Result<()> {
        let model_path = self.models_path.join(filename);
        
        if model_path.exists() {
            fs::remove_file(&model_path)
                .await
                .with_context(|| format!("Failed to delete model: {:?}", model_path))?;
            
            info!("Deleted model file: {:?}", model_path);
        }
        
        Ok(())
    }
    
    /// List all partial uploads (for cleanup purposes)
    pub async fn list_partial_uploads(&self) -> Result<Vec<String>> {
        let mut uploads = Vec::new();
        
        let mut entries = fs::read_dir(&self.base_path)
            .await
            .with_context(|| format!("Failed to read upload directory: {:?}", self.base_path))?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "part" {
                    if let Some(stem) = path.file_stem() {
                        uploads.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
        
        Ok(uploads)
    }
    
    /// Get the size of a partial upload
    pub async fn get_partial_size(&self, upload_id: &str) -> Result<u64> {
        let temp_path = self.base_path.join(format!("{}.part", upload_id));
        
        let metadata = fs::metadata(&temp_path)
            .await
            .with_context(|| format!("Failed to get metadata: {:?}", temp_path))?;
        
        Ok(metadata.len())
    }
    
    /// Check available disk space
    pub async fn available_space(&self) -> Result<u64> {
        // Use statvfs on Unix-like systems
        #[cfg(unix)]
        {
            
            
            let metadata = fs::metadata(&self.base_path)
                .await
                .context("Failed to get filesystem metadata")?;
            
            // This is a simplified check - in production you'd use statvfs
            // For now, we'll return a large value to not block uploads
            let _ = metadata;
            Ok(1024 * 1024 * 1024 * 1024) // 1TB placeholder
        }
        
        #[cfg(not(unix))]
        {
            Ok(1024 * 1024 * 1024 * 1024) // 1TB placeholder
        }
    }
}

impl Default for StorageManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn test_storage_manager() -> (StorageManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = StorageManager::with_paths(
            temp_dir.path().join("uploads"),
            temp_dir.path().join("models"),
        );
        storage.init().await.unwrap();
        (storage, temp_dir)
    }
    
    #[tokio::test]
    async fn test_create_temp_file() {
        let (storage, _temp) = test_storage_manager().await;
        
        let path = storage.create_temp_file("test_upload_123").await.unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("test_upload_123.part"));
    }
    
    #[tokio::test]
    async fn test_append_chunk_with_checksum() {
        let (storage, _temp) = test_storage_manager().await;
        
        storage.create_temp_file("test_upload").await.unwrap();
        storage.preallocate("test_upload", 1024).await.unwrap();
        
        let data = b"Hello, World!";
        let checksum = storage
            .append_chunk("test_upload", 0, 100, data, None)
            .await
            .unwrap();
        
        // Verify checksum is a valid hex string
        assert_eq!(checksum.len(), 64);
    }
    
    #[tokio::test]
    async fn test_checksum_verification() {
        let (storage, _temp) = test_storage_manager().await;
        
        storage.create_temp_file("test_upload").await.unwrap();
        storage.preallocate("test_upload", 1024).await.unwrap();
        
        let data = b"Test data";
        
        // First write to get the correct checksum
        let correct_checksum = storage
            .append_chunk("test_upload", 0, 100, data, None)
            .await
            .unwrap();
        
        // Should succeed with correct checksum
        let result = storage
            .append_chunk("test_upload", 0, 100, data, Some(&correct_checksum))
            .await;
        assert!(result.is_ok());
        
        // Should fail with wrong checksum
        let result = storage
            .append_chunk("test_upload", 0, 100, data, Some("wrong_checksum"))
            .await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_finalize_upload() {
        let (storage, _temp) = test_storage_manager().await;
        
        storage.create_temp_file("test_upload").await.unwrap();
        
        let final_path = storage.finalize("test_upload", "model.gguf").await.unwrap();
        
        assert!(final_path.exists());
        assert!(final_path.to_string_lossy().contains("model.gguf"));
    }
    
    #[tokio::test]
    async fn test_cleanup() {
        let (storage, _temp) = test_storage_manager().await;
        
        let temp_path = storage.create_temp_file("test_upload").await.unwrap();
        assert!(temp_path.exists());
        
        storage.cleanup("test_upload").await.unwrap();
        assert!(!temp_path.exists());
    }
}
