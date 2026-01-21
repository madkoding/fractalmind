//! Thread-safe LRU cache implementation.

#![allow(dead_code)]

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

use lru::LruCache;
use std::num::NonZeroUsize;

use super::config::{CacheConfig, CacheMetrics};
use super::entry::CacheEntry;

/// Thread-safe LRU cache with TTL support and metrics
pub struct ThreadSafeLruCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// The underlying LRU cache wrapped in a RwLock
    cache: RwLock<LruCache<K, CacheEntry<V>>>,

    /// Cache configuration
    config: CacheConfig,

    /// Atomic counters for metrics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
}

impl<K, V> ThreadSafeLruCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Creates a new cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        let capacity = NonZeroUsize::new(config.capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            cache: RwLock::new(LruCache::new(capacity)),
            config,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
        }
    }

    /// Creates a new cache with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(CacheConfig::with_capacity(capacity))
    }

    /// Gets a value from the cache
    ///
    /// Returns None if the key doesn't exist or the entry has expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().ok()?;

        // Check if key exists and peek at it first
        if let Some(entry) = cache.peek(key) {
            if entry.is_expired() {
                // Remove expired entry
                cache.pop(key);
                if self.config.track_metrics {
                    self.expirations.fetch_add(1, Ordering::Relaxed);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                }
                return None;
            }
        }

        // Now get the entry (which also promotes it in LRU order)
        if let Some(entry) = cache.get_mut(key) {
            entry.touch();
            if self.config.track_metrics {
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
            Some(entry.value.clone())
        } else {
            if self.config.track_metrics {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    }

    /// Puts a value into the cache
    ///
    /// Returns the previous value if it existed.
    pub fn put(&self, key: K, value: V) -> Option<V> {
        let entry = CacheEntry::new(value, self.config.ttl);
        self.put_entry(key, entry)
    }

    /// Puts a value with custom TTL
    pub fn put_with_ttl(&self, key: K, value: V, ttl: Duration) -> Option<V> {
        let entry = CacheEntry::new(value, Some(ttl));
        self.put_entry(key, entry)
    }

    /// Puts a permanent value (no TTL)
    pub fn put_permanent(&self, key: K, value: V) -> Option<V> {
        let entry = CacheEntry::permanent(value);
        self.put_entry(key, entry)
    }

    /// Internal method to put an entry
    fn put_entry(&self, key: K, entry: CacheEntry<V>) -> Option<V> {
        let mut cache = self.cache.write().ok()?;

        // Check if this will cause an eviction
        if cache.len() >= self.config.capacity && !cache.contains(&key) {
            if self.config.track_metrics {
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        cache.put(key, entry).map(|e| e.into_value())
    }

    /// Removes a value from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().ok()?;
        cache.pop(key).map(|e| e.into_value())
    }

    /// Checks if a key exists in the cache (without affecting LRU order)
    pub fn contains(&self, key: &K) -> bool {
        let cache = self.cache.read().ok();
        cache.map(|c| c.contains(key)).unwrap_or(false)
    }

    /// Returns the current number of entries
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Checks if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the capacity of the cache
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Clears all entries from the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Removes expired entries from the cache
    pub fn cleanup_expired(&self) -> usize {
        let mut cache = match self.cache.write() {
            Ok(c) => c,
            Err(_) => return 0,
        };

        // Collect expired keys
        let expired_keys: Vec<K> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        let count = expired_keys.len();

        // Remove expired entries
        for key in expired_keys {
            cache.pop(&key);
        }

        if self.config.track_metrics && count > 0 {
            self.expirations.fetch_add(count as u64, Ordering::Relaxed);
        }

        count
    }

    /// Returns the current cache metrics
    pub fn metrics(&self) -> CacheMetrics {
        CacheMetrics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
            size: self.len(),
        }
    }

    /// Resets all metrics counters
    pub fn reset_metrics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.expirations.store(0, Ordering::Relaxed);
    }

    /// Returns the configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Gets entry metadata without returning the value
    pub fn get_metadata(&self, key: &K) -> Option<EntryMetadata> {
        let cache = self.cache.read().ok()?;
        cache.peek(key).map(|entry| EntryMetadata {
            access_count: entry.access_count,
            age: entry.age(),
            idle_time: entry.idle_time(),
            remaining_ttl: entry.remaining_ttl(),
            is_expired: entry.is_expired(),
        })
    }
}

/// Metadata about a cache entry
#[derive(Debug, Clone)]
pub struct EntryMetadata {
    /// Number of times the entry has been accessed
    pub access_count: u64,

    /// How long since the entry was created
    pub age: Duration,

    /// How long since the entry was last accessed
    pub idle_time: Duration,

    /// Remaining TTL (None if no TTL or expired)
    pub remaining_ttl: Option<Duration>,

    /// Whether the entry has expired
    pub is_expired: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::{self, sleep};

    #[test]
    fn test_basic_operations() {
        let cache = ThreadSafeLruCache::<String, i32>::with_capacity(10);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.put("key1".to_string(), 100);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        let value = cache.get(&"key1".to_string());
        assert_eq!(value, Some(100));

        let missing = cache.get(&"missing".to_string());
        assert_eq!(missing, None);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = ThreadSafeLruCache::<i32, String>::with_capacity(3);

        cache.put(1, "one".to_string());
        cache.put(2, "two".to_string());
        cache.put(3, "three".to_string());

        // Access key 1 to make it recently used
        cache.get(&1);

        // Add key 4, which should evict key 2 (least recently used)
        cache.put(4, "four".to_string());

        assert!(cache.contains(&1));
        assert!(!cache.contains(&2)); // Should be evicted
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_ttl_expiration() {
        let config = CacheConfig::with_capacity(10).ttl(Duration::from_millis(50));
        let cache = ThreadSafeLruCache::<String, i32>::new(config);

        cache.put("key1".to_string(), 100);
        assert_eq!(cache.get(&"key1".to_string()), Some(100));

        sleep(Duration::from_millis(60));
        assert_eq!(cache.get(&"key1".to_string()), None);
    }

    #[test]
    fn test_custom_ttl_per_entry() {
        let config = CacheConfig::no_expiration(10);
        let cache = ThreadSafeLruCache::<String, i32>::new(config);

        cache.put_with_ttl("short".to_string(), 1, Duration::from_millis(50));
        cache.put_permanent("permanent".to_string(), 2);

        sleep(Duration::from_millis(60));

        assert_eq!(cache.get(&"short".to_string()), None);
        assert_eq!(cache.get(&"permanent".to_string()), Some(2));
    }

    #[test]
    fn test_remove() {
        let cache = ThreadSafeLruCache::<String, i32>::with_capacity(10);

        cache.put("key1".to_string(), 100);
        assert!(cache.contains(&"key1".to_string()));

        let removed = cache.remove(&"key1".to_string());
        assert_eq!(removed, Some(100));
        assert!(!cache.contains(&"key1".to_string()));

        let missing = cache.remove(&"missing".to_string());
        assert_eq!(missing, None);
    }

    #[test]
    fn test_clear() {
        let cache = ThreadSafeLruCache::<i32, i32>::with_capacity(10);

        cache.put(1, 1);
        cache.put(2, 2);
        cache.put(3, 3);

        assert_eq!(cache.len(), 3);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cleanup_expired() {
        let config = CacheConfig::with_capacity(10).ttl(Duration::from_millis(50));
        let cache = ThreadSafeLruCache::<i32, i32>::new(config);

        cache.put(1, 1);
        cache.put(2, 2);

        // Add one with longer TTL
        cache.put_with_ttl(3, 3, Duration::from_secs(60));

        sleep(Duration::from_millis(60));

        let cleaned = cache.cleanup_expired();
        assert_eq!(cleaned, 2); // Keys 1 and 2 should be expired
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&3));
    }

    #[test]
    fn test_metrics() {
        let config = CacheConfig::with_capacity(10).track_metrics(true);
        let cache = ThreadSafeLruCache::<String, i32>::new(config);

        // Put some values
        cache.put("key1".to_string(), 1);
        cache.put("key2".to_string(), 2);

        // Hits
        cache.get(&"key1".to_string());
        cache.get(&"key2".to_string());

        // Misses
        cache.get(&"missing1".to_string());
        cache.get(&"missing2".to_string());

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 2);
        assert_eq!(metrics.misses, 2);
        assert_eq!(metrics.size, 2);
        assert!((metrics.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let cache = Arc::new(ThreadSafeLruCache::<i32, i32>::with_capacity(100));
        let mut handles = vec![];

        // Spawn multiple threads doing reads and writes
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let key = i * 100 + j;
                    cache_clone.put(key, key * 2);
                    cache_clone.get(&key);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have some entries (not necessarily all due to eviction)
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_get_metadata() {
        let cache = ThreadSafeLruCache::<String, i32>::with_capacity(10);

        cache.put("key1".to_string(), 100);

        // Access a few times
        cache.get(&"key1".to_string());
        cache.get(&"key1".to_string());

        let metadata = cache.get_metadata(&"key1".to_string()).unwrap();
        assert_eq!(metadata.access_count, 3); // 1 put + 2 gets
        assert!(!metadata.is_expired);
    }

    #[test]
    fn test_update_existing() {
        let cache = ThreadSafeLruCache::<String, i32>::with_capacity(10);

        cache.put("key1".to_string(), 100);
        let old = cache.put("key1".to_string(), 200);

        assert_eq!(old, Some(100));
        assert_eq!(cache.get(&"key1".to_string()), Some(200));
        assert_eq!(cache.len(), 1);
    }
}
