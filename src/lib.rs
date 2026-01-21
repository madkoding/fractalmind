//! Fractal-Mind: Sistema de IA con memoria evolutiva y aprendizaje autónomo
//!
//! Este módulo expone la API pública de la biblioteca.

#![allow(dead_code)] // Permitir código no usado durante desarrollo

pub mod api;
pub mod cache;
pub mod db;
pub mod embeddings;
pub mod graph;
pub mod models;
pub mod services;
pub mod utils;

// Re-exportar tipos principales
pub use db::connection::{DatabaseConnection, DbConfig};
pub use models::llm::{BrainConfig, ModelBrain, ModelConfig, ModelProvider};
pub use models::node::{FractalNode, NodeMetadata, NodeStatus, NodeType, SourceType};
pub use models::edge::{FractalEdge, EdgeType, GraphPath};
pub use models::namespace::{Namespace, NamespaceType, Scope, ScopePermissions};
pub use models::embedding::{EmbeddingModel, EmbeddingVector};

// Embedding service exports
pub use embeddings::{
    EmbeddingConfig, EmbeddingDevice, EmbeddingProvider, EmbeddingResult,
    EmbeddingService, MockEmbeddingProvider,
};

// Cache exports
pub use cache::{
    CacheConfig, CacheMetrics, NodeCache, EmbeddingCache,
    SharedNodeCache, SharedEmbeddingCache, ThreadSafeLruCache,
};
