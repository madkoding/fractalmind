//! LRU Cache module for Fractal-Mind.
//!
//! This module provides optimized caching for frequently accessed data:
//! - Thread-safe LRU cache with TTL support
//! - Specialized caches for FractalNodes and embeddings
//! - Cache metrics for monitoring
//!
//! # Example
//!
//! ```rust,ignore
//! use fractalmind::cache::{CacheConfig, NodeCache, EmbeddingCache};
//!
//! // Create a node cache with 1000 entries
//! let node_cache = NodeCache::with_capacity(1000);
//!
//! // Create an embedding cache with custom config
//! let config = CacheConfig::with_capacity(5000)
//!     .ttl(std::time::Duration::from_secs(3600));
//! let embedding_cache = EmbeddingCache::new(config);
//! ```

pub mod config;
pub mod entry;
pub mod lru_cache;
pub mod node_cache;
pub mod embedding_cache;

// Re-exports
pub use config::{CacheConfig, CacheMetrics};
pub use entry::CacheEntry;
pub use lru_cache::{ThreadSafeLruCache, EntryMetadata};
pub use node_cache::{NodeCache, SharedNodeCache, new_shared_cache as new_shared_node_cache};
pub use embedding_cache::{EmbeddingCache, SharedEmbeddingCache, new_shared_cache as new_shared_embedding_cache};
