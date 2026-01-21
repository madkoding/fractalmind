//! Configuration for the cache module.

#![allow(dead_code)]

use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Configuration for the LRU cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub capacity: usize,

    /// Time-to-live for cache entries (None = no expiration)
    pub ttl: Option<Duration>,

    /// Whether to track cache metrics
    pub track_metrics: bool,

    /// Clean expired entries interval (for background cleanup)
    pub cleanup_interval: Option<Duration>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            ttl: Some(Duration::from_secs(3600)), // 1 hour default
            track_metrics: true,
            cleanup_interval: Some(Duration::from_secs(300)), // 5 minutes
        }
    }
}

impl CacheConfig {
    /// Creates a new cache configuration with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            ..Default::default()
        }
    }

    /// Creates a configuration without expiration
    pub fn no_expiration(capacity: usize) -> Self {
        Self {
            capacity,
            ttl: None,
            cleanup_interval: None,
            ..Default::default()
        }
    }

    /// Sets the TTL for cache entries
    pub fn ttl(mut self, duration: Duration) -> Self {
        self.ttl = Some(duration);
        self
    }

    /// Disables TTL
    pub fn no_ttl(mut self) -> Self {
        self.ttl = None;
        self
    }

    /// Sets whether to track metrics
    pub fn track_metrics(mut self, track: bool) -> Self {
        self.track_metrics = track;
        self
    }

    /// Sets the cleanup interval
    pub fn cleanup_interval(mut self, interval: Duration) -> Self {
        self.cleanup_interval = Some(interval);
        self
    }

    /// Loads configuration from environment variables
    pub fn from_env() -> Self {
        let capacity = std::env::var("CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        let ttl_secs = std::env::var("CACHE_TTL_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);

        let ttl = if ttl_secs > 0 {
            Some(Duration::from_secs(ttl_secs))
        } else {
            None
        };

        Self {
            capacity,
            ttl,
            ..Default::default()
        }
    }
}

/// Cache metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// Number of cache hits
    pub hits: u64,

    /// Number of cache misses
    pub misses: u64,

    /// Number of evictions due to capacity
    pub evictions: u64,

    /// Number of entries expired
    pub expirations: u64,

    /// Current number of entries
    pub size: usize,
}

impl CacheMetrics {
    /// Calculates hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Returns total requests (hits + misses)
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default();
        assert_eq!(config.capacity, 1000);
        assert!(config.ttl.is_some());
        assert!(config.track_metrics);
    }

    #[test]
    fn test_config_builder() {
        let config = CacheConfig::with_capacity(500)
            .ttl(Duration::from_secs(60))
            .track_metrics(false);

        assert_eq!(config.capacity, 500);
        assert_eq!(config.ttl, Some(Duration::from_secs(60)));
        assert!(!config.track_metrics);
    }

    #[test]
    fn test_no_expiration_config() {
        let config = CacheConfig::no_expiration(100);
        assert_eq!(config.capacity, 100);
        assert!(config.ttl.is_none());
    }

    #[test]
    fn test_metrics_hit_rate() {
        let mut metrics = CacheMetrics::default();
        assert_eq!(metrics.hit_rate(), 0.0);

        metrics.hits = 75;
        metrics.misses = 25;
        assert!((metrics.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = CacheMetrics {
            hits: 100,
            misses: 50,
            ..Default::default()
        };
        assert_eq!(metrics.total_requests(), 150);
    }
}
