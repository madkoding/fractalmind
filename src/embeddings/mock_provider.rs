//! Mock embedding provider for testing.

#![allow(dead_code)]

use anyhow::Result;
use async_trait::async_trait;
use std::time::Instant;

use crate::models::{EmbeddingModel, EmbeddingVector};
use super::provider::{BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult};

/// Mock embedding provider for testing
///
/// Generates deterministic embeddings based on text hash for reproducible tests.
pub struct MockEmbeddingProvider {
    model: EmbeddingModel,
    dimension: usize,
    latency_ms: u64,
    should_fail: bool,
}

impl MockEmbeddingProvider {
    /// Creates a new mock provider
    pub fn new(model: EmbeddingModel) -> Self {
        let dimension = model.dimension();
        Self {
            model,
            dimension: if dimension == 0 { 768 } else { dimension },
            latency_ms: 10,
            should_fail: false,
        }
    }

    /// Sets a custom dimension (useful for custom models)
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Sets the simulated latency
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Makes the provider fail on all operations
    pub fn should_fail(mut self, fail: bool) -> Self {
        self.should_fail = fail;
        self
    }

    /// Generates a deterministic embedding based on text hash
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // Use simple hash to generate deterministic values
        let hash = Self::simple_hash(text);
        let mut vector = Vec::with_capacity(self.dimension);

        for i in 0..self.dimension {
            // Generate pseudo-random value based on hash and position
            let value = ((hash.wrapping_add(i as u64).wrapping_mul(2654435761)) % 10000) as f32
                / 10000.0
                - 0.5;
            vector.push(value);
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        vector
    }

    /// Simple hash function for deterministic embeddings
    fn simple_hash(text: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        if self.should_fail {
            anyhow::bail!("Mock provider configured to fail");
        }

        let start = Instant::now();

        // Simulate latency
        if self.latency_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.latency_ms)).await;
        }

        let vector = self.generate_embedding(text);
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(EmbeddingResult {
            embedding: EmbeddingVector::new(vector, self.model.clone()),
            latency_ms,
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult> {
        if self.should_fail {
            anyhow::bail!("Mock provider configured to fail");
        }

        let start = Instant::now();

        // Simulate latency proportional to batch size
        if self.latency_ms > 0 {
            let total_latency = self.latency_ms * (texts.len() as u64).min(10);
            tokio::time::sleep(tokio::time::Duration::from_millis(total_latency)).await;
        }

        let embeddings: Vec<EmbeddingVector> = texts
            .iter()
            .map(|text| {
                let vector = self.generate_embedding(text);
                EmbeddingVector::new(vector, self.model.clone())
            })
            .collect();

        let count = embeddings.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(BatchEmbeddingResult {
            embeddings,
            latency_ms,
            count,
        })
    }

    fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(!self.should_fail)
    }

    fn provider_name(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_embed() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15)
            .with_latency(0);

        let result = provider.embed("hello world").await.unwrap();

        assert_eq!(result.embedding.dimension, 768);
        assert_eq!(result.embedding.vector.len(), 768);
        assert!(result.embedding.is_normalized());
    }

    #[tokio::test]
    async fn test_mock_provider_deterministic() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15)
            .with_latency(0);

        let result1 = provider.embed("hello world").await.unwrap();
        let result2 = provider.embed("hello world").await.unwrap();

        // Same text should produce same embedding
        assert_eq!(result1.embedding.vector, result2.embedding.vector);
    }

    #[tokio::test]
    async fn test_mock_provider_different_texts() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15)
            .with_latency(0);

        let result1 = provider.embed("hello world").await.unwrap();
        let result2 = provider.embed("goodbye world").await.unwrap();

        // Different texts should produce different embeddings
        assert_ne!(result1.embedding.vector, result2.embedding.vector);
    }

    #[tokio::test]
    async fn test_mock_provider_batch() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::BaaiGgeSmall)
            .with_latency(0);

        let texts = vec![
            "text one".to_string(),
            "text two".to_string(),
            "text three".to_string(),
        ];

        let result = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(result.count, 3);
        assert_eq!(result.embeddings.len(), 3);
        assert_eq!(result.embeddings[0].dimension, 384);
    }

    #[tokio::test]
    async fn test_mock_provider_should_fail() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15)
            .should_fail(true);

        let result = provider.embed("test").await;
        assert!(result.is_err());

        let batch_result = provider.embed_batch(&["test".to_string()]).await;
        assert!(batch_result.is_err());
    }

    #[tokio::test]
    async fn test_mock_provider_health_check() {
        let healthy_provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15);
        assert!(healthy_provider.health_check().await.unwrap());

        let unhealthy_provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15)
            .should_fail(true);
        assert!(!unhealthy_provider.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_mock_provider_custom_dimension() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::Custom("custom".to_string()))
            .with_dimension(1024)
            .with_latency(0);

        let result = provider.embed("test").await.unwrap();
        assert_eq!(result.embedding.dimension, 1024);
        assert_eq!(result.embedding.vector.len(), 1024);
    }

    #[tokio::test]
    async fn test_mock_provider_empty_batch() {
        let provider = MockEmbeddingProvider::new(EmbeddingModel::NomicEmbedTextV15)
            .with_latency(0);

        let result = provider.embed_batch(&[]).await.unwrap();
        assert_eq!(result.count, 0);
        assert!(result.embeddings.is_empty());
    }
}
