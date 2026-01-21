//! Embedding generation module for Fractal-Mind.
//!
//! This module provides text embedding capabilities using various providers:
//! - FastEmbed (local, requires `embeddings` feature)
//! - Mock provider (for testing)
//!
//! # Example
//!
//! ```rust,ignore
//! use fractalmind::embeddings::{EmbeddingConfig, EmbeddingService};
//! use fractalmind::models::EmbeddingModel;
//!
//! // Create service with mock provider (for testing)
//! let config = EmbeddingConfig::with_model(EmbeddingModel::NomicEmbedTextV15);
//! let service = EmbeddingService::with_mock(config);
//!
//! // Generate embedding
//! let result = service.embed("Hello world").await?;
//! println!("Embedding dimension: {}", result.embedding.dimension);
//! ```

pub mod config;
pub mod provider;
pub mod service;
pub mod mock_provider;

#[cfg(feature = "embeddings")]
pub mod fastembed_provider;

// Re-exports
pub use config::{EmbeddingConfig, EmbeddingDevice};
pub use provider::{BatchEmbeddingResult, EmbeddingProvider, EmbeddingProviderExt, EmbeddingResult};
pub use service::EmbeddingService;
pub use mock_provider::MockEmbeddingProvider;

#[cfg(feature = "embeddings")]
pub use fastembed_provider::FastEmbedProvider;
