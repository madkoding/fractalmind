//! Cache entry with metadata.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// A cache entry with metadata for TTL and access tracking
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// The cached value
    pub value: V,

    /// When the entry was created
    pub created_at: Instant,

    /// When the entry was last accessed
    pub last_accessed: Instant,

    /// Number of times this entry has been accessed
    pub access_count: u64,

    /// Time-to-live for this entry (None = no expiration)
    pub ttl: Option<Duration>,
}

impl<V> CacheEntry<V> {
    /// Creates a new cache entry
    pub fn new(value: V, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }

    /// Creates a cache entry without TTL
    pub fn permanent(value: V) -> Self {
        Self::new(value, None)
    }

    /// Checks if this entry has expired
    pub fn is_expired(&self) -> bool {
        match self.ttl {
            Some(ttl) => self.created_at.elapsed() > ttl,
            None => false,
        }
    }

    /// Marks the entry as accessed, updating metadata
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Returns the age of this entry
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns time since last access
    pub fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }

    /// Returns remaining TTL (None if no TTL or expired)
    pub fn remaining_ttl(&self) -> Option<Duration> {
        self.ttl.and_then(|ttl| {
            let elapsed = self.created_at.elapsed();
            if elapsed < ttl {
                Some(ttl - elapsed)
            } else {
                None
            }
        })
    }

    /// Updates the value and resets access time
    pub fn update(&mut self, value: V) {
        self.value = value;
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Consumes the entry and returns the value
    pub fn into_value(self) -> V {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_entry_creation() {
        let entry = CacheEntry::new("test value", Some(Duration::from_secs(60)));
        assert_eq!(entry.value, "test value");
        assert_eq!(entry.access_count, 1);
        assert!(entry.ttl.is_some());
    }

    #[test]
    fn test_permanent_entry() {
        let entry = CacheEntry::permanent(42);
        assert!(entry.ttl.is_none());
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_entry_expiration() {
        let entry = CacheEntry::new("test", Some(Duration::from_millis(10)));
        assert!(!entry.is_expired());

        sleep(Duration::from_millis(15));
        assert!(entry.is_expired());
    }

    #[test]
    fn test_entry_touch() {
        let mut entry = CacheEntry::new("test", None);
        assert_eq!(entry.access_count, 1);

        entry.touch();
        assert_eq!(entry.access_count, 2);

        entry.touch();
        assert_eq!(entry.access_count, 3);
    }

    #[test]
    fn test_entry_update() {
        let mut entry = CacheEntry::new("original", None);
        entry.update("updated");

        assert_eq!(entry.value, "updated");
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_remaining_ttl() {
        let entry = CacheEntry::new("test", Some(Duration::from_secs(60)));
        let remaining = entry.remaining_ttl().unwrap();
        assert!(remaining.as_secs() >= 59); // Allow some time to pass

        let expired_entry = CacheEntry::new("test", Some(Duration::from_millis(1)));
        sleep(Duration::from_millis(5));
        assert!(expired_entry.remaining_ttl().is_none());
    }

    #[test]
    fn test_into_value() {
        let entry = CacheEntry::new(vec![1, 2, 3], None);
        let value = entry.into_value();
        assert_eq!(value, vec![1, 2, 3]);
    }
}
