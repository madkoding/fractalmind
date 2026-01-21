//! Embedding service for generating text embeddings.

#![allow(dead_code)]

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::models::{EmbeddingModel, EmbeddingVector};
use super::config::EmbeddingConfig;
use super::provider::{BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult};
use super::mock_provider::MockEmbeddingProvider;

/// Main embedding service
///
/// Provides a high-level interface for generating embeddings with automatic
/// provider selection and fallback handling.
pub struct EmbeddingService {
    /// The embedding provider
    provider: Arc<dyn EmbeddingProvider>,

    /// Service configuration
    config: EmbeddingConfig,
}

impl EmbeddingService {
    /// Creates a new embedding service with the given provider
    pub fn new(provider: Arc<dyn EmbeddingProvider>, config: EmbeddingConfig) -> Self {
        info!(
            "EmbeddingService initialized with provider: {}, model: {:?}",
            provider.provider_name(),
            config.model
        );
        Self { provider, config }
    }

    /// Creates a new embedding service with a mock provider (for testing)
    pub fn with_mock(config: EmbeddingConfig) -> Self {
        let provider = MockEmbeddingProvider::new(config.model.clone());
        Self::new(Arc::new(provider), config)
    }

    /// Creates a new embedding service with FastEmbed provider
    #[cfg(feature = "embeddings")]
    pub fn with_fastembed(config: EmbeddingConfig) -> Result<Self> {
        use super::fastembed_provider::FastEmbedProvider;

        let provider = FastEmbedProvider::new(config.clone())
            .context("Failed to initialize FastEmbed provider")?;
        Ok(Self::new(Arc::new(provider), config))
    }

    /// Creates a service from environment configuration
    #[cfg(feature = "embeddings")]
    pub fn from_env() -> Result<Self> {
        let config = EmbeddingConfig::from_env();
        Self::with_fastembed(config)
    }

    /// Creates a mock service from environment configuration (for testing)
    pub fn mock_from_env() -> Self {
        let config = EmbeddingConfig::from_env();
        Self::with_mock(config)
    }

    /// Generates an embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        if text.is_empty() {
            anyhow::bail!("Cannot generate embedding for empty text");
        }

        debug!("Generating embedding for text ({} chars)", text.len());
        self.provider.embed(text).await
    }

    /// Generates embeddings for multiple texts
    pub async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResult {
                embeddings: vec![],
                latency_ms: 0,
                count: 0,
            });
        }

        // Filter out empty texts
        let non_empty_texts: Vec<&String> = texts.iter().filter(|t| !t.is_empty()).collect();
        if non_empty_texts.len() != texts.len() {
            warn!(
                "Filtered {} empty texts from batch of {}",
                texts.len() - non_empty_texts.len(),
                texts.len()
            );
        }

        if non_empty_texts.is_empty() {
            return Ok(BatchEmbeddingResult {
                embeddings: vec![],
                latency_ms: 0,
                count: 0,
            });
        }

        let owned_texts: Vec<String> = non_empty_texts.into_iter().cloned().collect();
        debug!("Generating batch embeddings for {} texts", owned_texts.len());
        self.provider.embed_batch(&owned_texts).await
    }

    /// Generates an embedding and returns only the vector
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let result = self.embed(text).await?;
        Ok(result.embedding.vector)
    }

    /// Generates an EmbeddingVector (compatible with node model)
    pub async fn embed_to_vector(&self, text: &str) -> Result<EmbeddingVector> {
        let result = self.embed(text).await?;
        Ok(result.embedding)
    }

    /// Returns the current model
    pub fn model(&self) -> &EmbeddingModel {
        self.provider.model()
    }

    /// Returns the embedding dimension
    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    /// Returns the provider name
    pub fn provider_name(&self) -> &str {
        self.provider.provider_name()
    }

    /// Checks if the service is healthy
    pub async fn health_check(&self) -> Result<bool> {
        self.provider.health_check().await
    }

    /// Returns the configuration
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_service() -> EmbeddingService {
        let config = EmbeddingConfig::with_model(EmbeddingModel::NomicEmbedTextV15);
        EmbeddingService::with_mock(config)
    }

    #[tokio::test]
    async fn test_embed_single_text() {
        let service = create_test_service();
        let result = service.embed("hello world").await.unwrap();

        assert_eq!(result.embedding.dimension, 768);
        assert!(result.embedding.is_normalized());
    }

    #[tokio::test]
    async fn test_embed_empty_text_fails() {
        let service = create_test_service();
        let result = service.embed("").await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty text"));
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let service = create_test_service();
        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];

        let result = service.embed_batch(&texts).await.unwrap();

        assert_eq!(result.count, 3);
        assert_eq!(result.embeddings.len(), 3);
    }

    #[tokio::test]
    async fn test_embed_batch_filters_empty() {
        let service = create_test_service();
        let texts = vec![
            "first text".to_string(),
            "".to_string(),
            "third text".to_string(),
        ];

        let result = service.embed_batch(&texts).await.unwrap();

        // Empty text should be filtered out
        assert_eq!(result.count, 2);
        assert_eq!(result.embeddings.len(), 2);
    }

    #[tokio::test]
    async fn test_embed_batch_empty() {
        let service = create_test_service();
        let texts: Vec<String> = vec![];

        let result = service.embed_batch(&texts).await.unwrap();

        assert_eq!(result.count, 0);
        assert!(result.embeddings.is_empty());
    }

    #[tokio::test]
    async fn test_embed_text_returns_vector() {
        let service = create_test_service();
        let vector = service.embed_text("hello world").await.unwrap();

        assert_eq!(vector.len(), 768);
    }

    #[tokio::test]
    async fn test_embed_to_vector() {
        let service = create_test_service();
        let embedding = service.embed_to_vector("hello world").await.unwrap();

        assert_eq!(embedding.model, EmbeddingModel::NomicEmbedTextV15);
        assert_eq!(embedding.dimension, 768);
    }

    #[tokio::test]
    async fn test_service_metadata() {
        let service = create_test_service();

        assert_eq!(*service.model(), EmbeddingModel::NomicEmbedTextV15);
        assert_eq!(service.dimension(), 768);
        assert_eq!(service.provider_name(), "mock");
    }

    #[tokio::test]
    async fn test_health_check() {
        let service = create_test_service();
        assert!(service.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_service_with_different_model() {
        let config = EmbeddingConfig::with_model(EmbeddingModel::BaaiGgeSmall);
        let service = EmbeddingService::with_mock(config);

        assert_eq!(*service.model(), EmbeddingModel::BaaiGgeSmall);
        assert_eq!(service.dimension(), 384);

        let result = service.embed("test").await.unwrap();
        assert_eq!(result.embedding.dimension, 384);
    }

    #[tokio::test]
    async fn test_mock_from_env() {
        // This uses default config since env vars likely aren't set
        let service = EmbeddingService::mock_from_env();
        assert_eq!(service.provider_name(), "mock");
    }
}
