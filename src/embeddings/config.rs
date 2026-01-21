//! Configuration for the embedding service.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use crate::models::EmbeddingModel;

/// Configuration for the embedding service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// The embedding model to use
    pub model: EmbeddingModel,

    /// Maximum batch size for embedding generation
    pub batch_size: usize,

    /// Whether to normalize vectors after generation
    pub normalize: bool,

    /// Cache directory for downloaded models
    pub cache_dir: Option<String>,

    /// Device to use (cpu, cuda, etc.)
    pub device: EmbeddingDevice,
}

/// Device configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingDevice {
    #[default]
    Cpu,
    Cuda,
    CudaDevice(usize),
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::NomicEmbedTextV15,
            batch_size: 32,
            normalize: true,
            cache_dir: None,
            device: EmbeddingDevice::default(),
        }
    }
}

impl EmbeddingConfig {
    /// Creates a new configuration with the specified model
    pub fn with_model(model: EmbeddingModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Sets the batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Sets whether to normalize vectors
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets the cache directory
    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Sets the device
    pub fn device(mut self, device: EmbeddingDevice) -> Self {
        self.device = device;
        self
    }

    /// Loads configuration from environment variables
    pub fn from_env() -> Self {
        let model = std::env::var("EMBEDDING_MODEL")
            .map(|m| match m.as_str() {
                "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
                "bge-small-en-v1.5" => EmbeddingModel::BaaiGgeSmall,
                "all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLmL6V2,
                "clip-vit-base-patch32" => EmbeddingModel::ClipVitB32,
                other => EmbeddingModel::Custom(other.to_string()),
            })
            .unwrap_or(EmbeddingModel::NomicEmbedTextV15);

        let batch_size = std::env::var("EMBEDDING_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);

        let normalize = std::env::var("EMBEDDING_NORMALIZE")
            .map(|s| s == "true" || s == "1")
            .unwrap_or(true);

        let cache_dir = std::env::var("EMBEDDING_CACHE_DIR").ok();

        let device = std::env::var("EMBEDDING_DEVICE")
            .map(|d| match d.as_str() {
                "cuda" => EmbeddingDevice::Cuda,
                "cpu" => EmbeddingDevice::Cpu,
                d if d.starts_with("cuda:") => {
                    d[5..].parse().ok()
                        .map(EmbeddingDevice::CudaDevice)
                        .unwrap_or(EmbeddingDevice::Cuda)
                }
                _ => EmbeddingDevice::Cpu,
            })
            .unwrap_or(EmbeddingDevice::Cpu);

        Self {
            model,
            batch_size,
            normalize,
            cache_dir,
            device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model, EmbeddingModel::NomicEmbedTextV15);
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
        assert!(config.cache_dir.is_none());
        assert_eq!(config.device, EmbeddingDevice::Cpu);
    }

    #[test]
    fn test_config_builder() {
        let config = EmbeddingConfig::with_model(EmbeddingModel::BaaiGgeSmall)
            .batch_size(64)
            .normalize(false)
            .cache_dir("/tmp/embeddings")
            .device(EmbeddingDevice::Cuda);

        assert_eq!(config.model, EmbeddingModel::BaaiGgeSmall);
        assert_eq!(config.batch_size, 64);
        assert!(!config.normalize);
        assert_eq!(config.cache_dir, Some("/tmp/embeddings".to_string()));
        assert_eq!(config.device, EmbeddingDevice::Cuda);
    }
}
