//! Specialized LRU cache for FractalNodes.

#![allow(dead_code)]

use std::sync::Arc;
use surrealdb::sql::Thing;
use tracing::{debug, info};

use crate::models::FractalNode;
use super::config::{CacheConfig, CacheMetrics};
use super::lru_cache::ThreadSafeLruCache;

/// Specialized cache for FractalNodes
///
/// Uses the node's database ID as the cache key.
pub struct NodeCache {
    /// Internal LRU cache
    cache: ThreadSafeLruCache<String, FractalNode>,
}

impl NodeCache {
    /// Creates a new node cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        info!("Initializing NodeCache with capacity: {}", config.capacity);
        Self {
            cache: ThreadSafeLruCache::new(config),
        }
    }

    /// Creates a new node cache with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(CacheConfig::with_capacity(capacity))
    }

    /// Creates from environment configuration
    pub fn from_env() -> Self {
        Self::new(CacheConfig::from_env())
    }

    /// Gets a node by its database ID
    pub fn get(&self, id: &Thing) -> Option<FractalNode> {
        let key = Self::thing_to_key(id);
        debug!("Cache lookup for node: {}", key);
        self.cache.get(&key)
    }

    /// Gets a node by string key
    pub fn get_by_key(&self, key: &str) -> Option<FractalNode> {
        self.cache.get(&key.to_string())
    }

    /// Puts a node into the cache
    ///
    /// Uses the node's ID as the key. Returns false if the node has no ID.
    pub fn put(&self, node: &FractalNode) -> bool {
        if let Some(ref id) = node.id {
            let key = Self::thing_to_key(id);
            debug!("Caching node: {}", key);
            self.cache.put(key, node.clone());
            true
        } else {
            debug!("Cannot cache node without ID");
            false
        }
    }

    /// Puts a node with a custom key
    pub fn put_with_key(&self, key: String, node: FractalNode) {
        self.cache.put(key, node);
    }

    /// Puts multiple nodes into the cache
    ///
    /// Returns the number of nodes successfully cached.
    pub fn put_batch(&self, nodes: &[FractalNode]) -> usize {
        let mut count = 0;
        for node in nodes {
            if self.put(node) {
                count += 1;
            }
        }
        debug!("Batch cached {} nodes", count);
        count
    }

    /// Removes a node from the cache by ID
    pub fn remove(&self, id: &Thing) -> Option<FractalNode> {
        let key = Self::thing_to_key(id);
        self.cache.remove(&key)
    }

    /// Checks if a node is in the cache
    pub fn contains(&self, id: &Thing) -> bool {
        let key = Self::thing_to_key(id);
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
        info!("Clearing node cache");
        self.cache.clear();
    }

    /// Removes expired entries
    pub fn cleanup_expired(&self) -> usize {
        let count = self.cache.cleanup_expired();
        if count > 0 {
            debug!("Cleaned up {} expired node cache entries", count);
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

    /// Gets the underlying cache (for advanced operations)
    pub fn inner(&self) -> &ThreadSafeLruCache<String, FractalNode> {
        &self.cache
    }

    /// Converts a Thing ID to a cache key
    fn thing_to_key(thing: &Thing) -> String {
        format!("{}:{}", thing.tb, thing.id)
    }
}

/// Thread-safe wrapper for NodeCache
pub type SharedNodeCache = Arc<NodeCache>;

/// Creates a new shared node cache
pub fn new_shared_cache(config: CacheConfig) -> SharedNodeCache {
    Arc::new(NodeCache::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EmbeddingModel, EmbeddingVector, NodeMetadata};
    use surrealdb::sql::Id;

    fn create_test_node(id: &str) -> FractalNode {
        let embedding = EmbeddingVector::new(
            vec![0.1; 768],
            EmbeddingModel::NomicEmbedTextV15,
        );

        let mut node = FractalNode::new_leaf(
            format!("Content for node {}", id),
            embedding,
            "test_namespace".to_string(),
            None,
            NodeMetadata::default(),
        );

        node.id = Some(Thing {
            tb: "nodes".to_string(),
            id: Id::String(id.to_string()),
        });

        node
    }

    #[test]
    fn test_node_cache_basic() {
        let cache = NodeCache::with_capacity(10);
        assert!(cache.is_empty());

        let node = create_test_node("1");
        assert!(cache.put(&node));

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let id = Thing {
            tb: "nodes".to_string(),
            id: Id::String("1".to_string()),
        };
        assert!(cache.contains(&id));

        let retrieved = cache.get(&id).unwrap();
        assert!(retrieved.content.contains("node 1"));
    }

    #[test]
    fn test_node_cache_without_id() {
        let cache = NodeCache::with_capacity(10);

        let embedding = EmbeddingVector::new(
            vec![0.1; 768],
            EmbeddingModel::NomicEmbedTextV15,
        );

        let node = FractalNode::new_leaf(
            "No ID node".to_string(),
            embedding,
            "test".to_string(),
            None,
            NodeMetadata::default(),
        );

        // Node without ID should not be cached
        assert!(!cache.put(&node));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_node_cache_batch() {
        let cache = NodeCache::with_capacity(10);

        let nodes: Vec<FractalNode> = (1..=5)
            .map(|i| create_test_node(&i.to_string()))
            .collect();

        let cached = cache.put_batch(&nodes);
        assert_eq!(cached, 5);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_node_cache_remove() {
        let cache = NodeCache::with_capacity(10);
        let node = create_test_node("1");
        cache.put(&node);

        let id = Thing {
            tb: "nodes".to_string(),
            id: Id::String("1".to_string()),
        };

        let removed = cache.remove(&id);
        assert!(removed.is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_node_cache_clear() {
        let cache = NodeCache::with_capacity(10);

        for i in 1..=5 {
            let node = create_test_node(&i.to_string());
            cache.put(&node);
        }

        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_node_cache_metrics() {
        let cache = NodeCache::with_capacity(10);
        let node = create_test_node("1");
        cache.put(&node);

        let id = Thing {
            tb: "nodes".to_string(),
            id: Id::String("1".to_string()),
        };

        // Hit
        cache.get(&id);

        // Miss
        let missing_id = Thing {
            tb: "nodes".to_string(),
            id: Id::String("999".to_string()),
        };
        cache.get(&missing_id);

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
    }

    #[test]
    fn test_node_cache_eviction() {
        let cache = NodeCache::with_capacity(3);

        for i in 1..=4 {
            let node = create_test_node(&i.to_string());
            cache.put(&node);
        }

        // Should have evicted the first node
        assert_eq!(cache.len(), 3);

        let id1 = Thing {
            tb: "nodes".to_string(),
            id: Id::String("1".to_string()),
        };
        assert!(!cache.contains(&id1));

        let id4 = Thing {
            tb: "nodes".to_string(),
            id: Id::String("4".to_string()),
        };
        assert!(cache.contains(&id4));
    }

    #[test]
    fn test_shared_cache() {
        let cache = new_shared_cache(CacheConfig::with_capacity(10));
        let node = create_test_node("1");
        cache.put(&node);

        // Clone the Arc and verify access
        let cache2 = Arc::clone(&cache);
        let id = Thing {
            tb: "nodes".to_string(),
            id: Id::String("1".to_string()),
        };
        assert!(cache2.get(&id).is_some());
    }
}
