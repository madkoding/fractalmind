//! FastEmbed-based embedding provider.
//!
//! This module requires the `embeddings` feature to be enabled.

#![cfg(feature = "embeddings")]

use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::models::{EmbeddingModel, EmbeddingVector};
use super::config::EmbeddingConfig;
use super::provider::{BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult};

/// FastEmbed-based embedding provider
pub struct FastEmbedProvider {
    /// The text embedding model
    model: Arc<RwLock<TextEmbedding>>,

    /// Configuration
    config: EmbeddingConfig,

    /// Cached dimension value
    dimension: usize,
}

impl FastEmbedProvider {
    /// Creates a new FastEmbed provider with the given configuration
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        info!("Initializing FastEmbed provider with model: {:?}", config.model);

        let fastembed_model = Self::map_model(&config.model)?;
        let dimension = config.model.dimension();

        let mut init_options = InitOptions::new(fastembed_model);

        if let Some(ref cache_dir) = config.cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.into());
        }

        let model = TextEmbedding::try_new(init_options)
            .context("Failed to initialize FastEmbed model")?;

        info!("FastEmbed provider initialized successfully (dimension: {})", dimension);

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
            dimension,
        })
    }

    /// Maps our EmbeddingModel enum to FastEmbed's model enum
    fn map_model(model: &EmbeddingModel) -> Result<FastEmbedModel> {
        match model {
            EmbeddingModel::NomicEmbedTextV15 => Ok(FastEmbedModel::NomicEmbedTextV15),
            EmbeddingModel::BaaiGgeSmall => Ok(FastEmbedModel::BGESmallENV15),
            EmbeddingModel::AllMiniLmL6V2 => Ok(FastEmbedModel::AllMiniLML6V2),
            EmbeddingModel::ClipVitB32 => {
                anyhow::bail!("CLIP model not supported by FastEmbed text embedding")
            }
            EmbeddingModel::Custom(name) => {
                anyhow::bail!("Custom model '{}' not supported by FastEmbed", name)
            }
        }
    }

    /// Normalizes a vector to unit length
    fn normalize_vector(mut vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        let start = Instant::now();
        let text_owned = text.to_string();
        let model = self.config.model.clone();
        let normalize = self.config.normalize;

        debug!("Generating embedding for text ({} chars)", text.len());

        let model_lock = self.model.read().await;
        let embeddings = model_lock
            .embed(vec![text_owned], None)
            .context("Failed to generate embedding")?;

        let mut vector = embeddings
            .into_iter()
            .next()
            .context("No embedding returned")?;

        if normalize {
            vector = Self::normalize_vector(vector);
        }

        let latency_ms = start.elapsed().as_millis() as u64;

        debug!("Embedding generated in {}ms", latency_ms);

        Ok(EmbeddingResult {
            embedding: EmbeddingVector::new(vector, model),
            latency_ms,
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResult {
                embeddings: vec![],
                latency_ms: 0,
                count: 0,
            });
        }

        let start = Instant::now();
        let model = self.config.model.clone();
        let normalize = self.config.normalize;
        let batch_size = self.config.batch_size;

        debug!("Generating batch embeddings for {} texts (batch_size: {})",
               texts.len(), batch_size);

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in texts.chunks(batch_size) {
            let model_lock = self.model.read().await;
            let embeddings = model_lock
                .embed(chunk.to_vec(), None)
                .context("Failed to generate batch embeddings")?;

            for mut vector in embeddings {
                if normalize {
                    vector = Self::normalize_vector(vector);
                }
                all_embeddings.push(EmbeddingVector::new(vector, model.clone()));
            }
        }

        let latency_ms = start.elapsed().as_millis() as u64;
        let count = all_embeddings.len();

        debug!("Batch embedding generated {} vectors in {}ms", count, latency_ms);

        Ok(BatchEmbeddingResult {
            embeddings: all_embeddings,
            latency_ms,
            count,
        })
    }

    fn model(&self) -> &EmbeddingModel {
        &self.config.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn health_check(&self) -> Result<bool> {
        // Try to generate a simple embedding to verify the model is working
        let result = self.embed("health check").await;
        Ok(result.is_ok())
    }

    fn provider_name(&self) -> &str {
        "fastembed"
    }
}

#[cfg(test)]
mod tests {
    // Tests require the model to be downloaded, so they're integration tests
    // Unit tests would require mocking the TextEmbedding struct
}
