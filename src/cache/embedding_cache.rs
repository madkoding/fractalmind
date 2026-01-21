//! Specialized LRU cache for embedding vectors.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

use crate::models::EmbeddingVector;
use super::config::{CacheConfig, CacheMetrics};
use super::lru_cache::ThreadSafeLruCache;

/// Specialized cache for embedding vectors
///
/// Caches embeddings by their text content hash to avoid recomputing
/// embeddings for frequently accessed texts.
pub struct EmbeddingCache {
    /// Internal LRU cache
    cache: ThreadSafeLruCache<u64, EmbeddingVector>,
}

impl EmbeddingCache {
    /// Creates a new embedding cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        info!("Initializing EmbeddingCache with capacity: {}", config.capacity);
        Self {
            cache: ThreadSafeLruCache::new(config),
        }
    }

    /// Creates a new embedding cache with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(CacheConfig::with_capacity(capacity))
    }

    /// Creates with default embedding cache settings
    ///
    /// Embeddings are relatively stable, so use a longer TTL.
    pub fn default_config() -> Self {
        let config = CacheConfig::with_capacity(5000)
            .ttl(Duration::from_secs(86400)) // 24 hours
            .track_metrics(true);
        Self::new(config)
    }

    /// Gets an embedding by text
    pub fn get(&self, text: &str) -> Option<EmbeddingVector> {
        let key = Self::hash_text(text);
        debug!("Embedding cache lookup for hash: {}", key);
        self.cache.get(&key)
    }

    /// Gets an embedding by pre-computed hash
    pub fn get_by_hash(&self, hash: u64) -> Option<EmbeddingVector> {
        self.cache.get(&hash)
    }

    /// Puts an embedding into the cache
    pub fn put(&self, text: &str, embedding: EmbeddingVector) {
        let key = Self::hash_text(text);
        debug!("Caching embedding for hash: {}", key);
        self.cache.put(key, embedding);
    }

    /// Puts an embedding with a pre-computed hash
    pub fn put_by_hash(&self, hash: u64, embedding: EmbeddingVector) {
        self.cache.put(hash, embedding);
    }

    /// Puts multiple embeddings into the cache
    pub fn put_batch(&self, texts: &[String], embeddings: &[EmbeddingVector]) {
        if texts.len() != embeddings.len() {
            debug!("Batch put: texts and embeddings have different lengths");
            return;
        }

        for (text, embedding) in texts.iter().zip(embeddings.iter()) {
            self.put(text, embedding.clone());
        }
    }

    /// Removes an embedding from the cache
    pub fn remove(&self, text: &str) -> Option<EmbeddingVector> {
        let key = Self::hash_text(text);
        self.cache.remove(&key)
    }

    /// Checks if an embedding is cached for the given text
    pub fn contains(&self, text: &str) -> bool {
        let key = Self::hash_text(text);
        self.cache.contains(&key)
    }

    /// Returns current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Checks if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Returns cache capacity
    pub fn capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Clears all entries
    pub fn clear(&self) {
        info!("Clearing embedding cache");
        self.cache.clear();
    }

    /// Removes expired entries
    pub fn cleanup_expired(&self) -> usize {
        let count = self.cache.cleanup_expired();
        if count > 0 {
            debug!("Cleaned up {} expired embedding cache entries", count);
        }
        count
    }

    /// Returns cache metrics
    pub fn metrics(&self) -> CacheMetrics {
        self.cache.metrics()
    }

    /// Resets metrics counters
    pub fn reset_metrics(&self) {
        self.cache.reset_metrics();
    }

    /// Computes a hash for the given text
    ///
    /// Uses a fast, stable hash function suitable for cache keys.
    pub fn hash_text(text: &str) -> u64 {
        // Using FNV-1a hash for speed and good distribution
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;

        let mut hash = FNV_OFFSET;
        for byte in text.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Gets the underlying cache
    pub fn inner(&self) -> &ThreadSafeLruCache<u64, EmbeddingVector> {
        &self.cache
    }
}

/// Thread-safe wrapper for EmbeddingCache
pub type SharedEmbeddingCache = Arc<EmbeddingCache>;

/// Creates a new shared embedding cache
pub fn new_shared_cache(config: CacheConfig) -> SharedEmbeddingCache {
    Arc::new(EmbeddingCache::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EmbeddingModel;

    fn create_test_embedding(value: f32) -> EmbeddingVector {
        EmbeddingVector::new(
            vec![value; 768],
            EmbeddingModel::NomicEmbedTextV15,
        )
    }

    #[test]
    fn test_embedding_cache_basic() {
        let cache = EmbeddingCache::with_capacity(10);
        assert!(cache.is_empty());

        let embedding = create_test_embedding(0.5);
        cache.put("hello world", embedding.clone());

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let retrieved = cache.get("hello world").unwrap();
        assert_eq!(retrieved.vector, embedding.vector);
    }

    #[test]
    fn test_embedding_cache_miss() {
        let cache = EmbeddingCache::with_capacity(10);
        let embedding = create_test_embedding(0.5);
        cache.put("hello", embedding);

        // Different text should miss
        assert!(cache.get("world").is_none());
    }

    #[test]
    fn test_hash_consistency() {
        let hash1 = EmbeddingCache::hash_text("hello world");
        let hash2 = EmbeddingCache::hash_text("hello world");
        let hash3 = EmbeddingCache::hash_text("different text");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_batch_put() {
        let cache = EmbeddingCache::with_capacity(10);

        let texts = vec!["text1".to_string(), "text2".to_string(), "text3".to_string()];
        let embeddings: Vec<EmbeddingVector> = (1..=3)
            .map(|i| create_test_embedding(i as f32 * 0.1))
            .collect();

        cache.put_batch(&texts, &embeddings);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains("text1"));
        assert!(cache.contains("text2"));
        assert!(cache.contains("text3"));
    }

    #[test]
    fn test_remove() {
        let cache = EmbeddingCache::with_capacity(10);
        let embedding = create_test_embedding(0.5);
        cache.put("hello", embedding.clone());

        let removed = cache.remove("hello");
        assert!(removed.is_some());
        assert!(!cache.contains("hello"));
    }

    #[test]
    fn test_clear() {
        let cache = EmbeddingCache::with_capacity(10);

        for i in 0..5 {
            let embedding = create_test_embedding(i as f32 * 0.1);
            cache.put(&format!("text{}", i), embedding);
        }

        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_metrics() {
        let cache = EmbeddingCache::with_capacity(10);
        let embedding = create_test_embedding(0.5);
        cache.put("hello", embedding);

        // Hit
        cache.get("hello");

        // Miss
        cache.get("missing");

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
    }

    #[test]
    fn test_eviction() {
        let cache = EmbeddingCache::with_capacity(3);

        for i in 0..4 {
            let embedding = create_test_embedding(i as f32 * 0.1);
            cache.put(&format!("text{}", i), embedding);
        }

        // Should have evicted the first entry
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains("text0"));
        assert!(cache.contains("text3"));
    }

    #[test]
    fn test_default_config() {
        let cache = EmbeddingCache::default_config();
        assert_eq!(cache.capacity(), 5000);
    }

    #[test]
    fn test_shared_cache() {
        let cache = new_shared_cache(CacheConfig::with_capacity(10));
        let embedding = create_test_embedding(0.5);
        cache.put("hello", embedding);

        let cache2 = Arc::clone(&cache);
        assert!(cache2.get("hello").is_some());
    }

    #[test]
    fn test_get_by_hash() {
        let cache = EmbeddingCache::with_capacity(10);
        let embedding = create_test_embedding(0.5);

        let hash = EmbeddingCache::hash_text("hello");
        cache.put("hello", embedding.clone());

        let retrieved = cache.get_by_hash(hash).unwrap();
        assert_eq!(retrieved.vector, embedding.vector);
    }
}
