//! Embedding provider trait for abstraction over different backends.

#![allow(dead_code)]

use anyhow::Result;
use async_trait::async_trait;
use crate::models::{EmbeddingModel, EmbeddingVector};

/// Result of an embedding operation
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The generated embedding vector
    pub embedding: EmbeddingVector,

    /// Processing time in milliseconds
    pub latency_ms: u64,
}

/// Batch embedding result
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResult {
    /// The generated embedding vectors
    pub embeddings: Vec<EmbeddingVector>,

    /// Total processing time in milliseconds
    pub latency_ms: u64,

    /// Number of texts processed
    pub count: usize,
}

/// Trait for embedding providers
///
/// This trait abstracts over different embedding backends like FastEmbed,
/// Ollama, OpenAI, etc. allowing for easy switching between providers.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generates an embedding for a single text
    async fn embed(&self, text: &str) -> Result<EmbeddingResult>;

    /// Generates embeddings for multiple texts in batch
    async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult>;

    /// Returns the embedding model being used
    fn model(&self) -> &EmbeddingModel;

    /// Returns the dimension of the generated embeddings
    fn dimension(&self) -> usize;

    /// Checks if the provider is healthy and ready
    async fn health_check(&self) -> Result<bool>;

    /// Returns the provider name for logging/metrics
    fn provider_name(&self) -> &str;
}

/// Extension trait for EmbeddingProvider with utility methods
#[async_trait]
pub trait EmbeddingProviderExt: EmbeddingProvider {
    /// Embeds a text and returns only the vector (convenience method)
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let result = self.embed(text).await?;
        Ok(result.embedding.vector)
    }

    /// Embeds multiple texts and returns only the vectors
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let result = self.embed_batch(texts).await?;
        Ok(result.embeddings.into_iter().map(|e| e.vector).collect())
    }
}

// Blanket implementation for all EmbeddingProvider implementations
impl<T: EmbeddingProvider> EmbeddingProviderExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_result_creation() {
        let embedding = EmbeddingVector::new(
            vec![0.1, 0.2, 0.3],
            EmbeddingModel::NomicEmbedTextV15,
        );
        let result = EmbeddingResult {
            embedding,
            latency_ms: 100,
        };

        assert_eq!(result.latency_ms, 100);
        assert_eq!(result.embedding.vector.len(), 3);
    }

    #[test]
    fn test_batch_embedding_result_creation() {
        let embeddings = vec![
            EmbeddingVector::new(vec![0.1, 0.2], EmbeddingModel::BaaiGgeSmall),
            EmbeddingVector::new(vec![0.3, 0.4], EmbeddingModel::BaaiGgeSmall),
        ];
        let result = BatchEmbeddingResult {
            embeddings,
            latency_ms: 200,
            count: 2,
        };

        assert_eq!(result.count, 2);
        assert_eq!(result.latency_ms, 200);
        assert_eq!(result.embeddings.len(), 2);
    }
}
