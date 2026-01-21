//! Service configuration.

#![allow(dead_code)]

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration for the REM phase service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemPhaseConfig {
    /// Interval between REM phase runs.
    pub interval: Duration,

    /// Maximum number of nodes to process per run.
    pub max_nodes_per_run: usize,

    /// Whether to enable web search for incomplete nodes.
    pub enable_web_search: bool,

    /// Maximum number of web search results per query.
    pub max_search_results: usize,

    /// Whether to enable RAPTOR clustering.
    pub enable_clustering: bool,

    /// Minimum similarity threshold for creating cross-namespace links.
    pub cross_namespace_threshold: f32,

    /// Whether to run in the background automatically.
    pub auto_run: bool,

    /// Timeout for web search operations.
    pub search_timeout: Duration,

    /// Batch size for processing nodes.
    pub batch_size: usize,
}

impl Default for RemPhaseConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3600), // 1 hour
            max_nodes_per_run: 100,
            enable_web_search: true,
            max_search_results: 5,
            enable_clustering: true,
            cross_namespace_threshold: 0.7,
            auto_run: false,
            search_timeout: Duration::from_secs(30),
            batch_size: 10,
        }
    }
}

impl RemPhaseConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set interval.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Builder: set max nodes per run.
    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes_per_run = max.max(1);
        self
    }

    /// Builder: enable/disable web search.
    pub fn with_web_search(mut self, enabled: bool) -> Self {
        self.enable_web_search = enabled;
        self
    }

    /// Builder: set max search results.
    pub fn with_max_search_results(mut self, max: usize) -> Self {
        self.max_search_results = max.max(1);
        self
    }

    /// Builder: enable/disable clustering.
    pub fn with_clustering(mut self, enabled: bool) -> Self {
        self.enable_clustering = enabled;
        self
    }

    /// Builder: set cross-namespace threshold.
    pub fn with_cross_namespace_threshold(mut self, threshold: f32) -> Self {
        self.cross_namespace_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Builder: enable/disable auto run.
    pub fn with_auto_run(mut self, enabled: bool) -> Self {
        self.auto_run = enabled;
        self
    }

    /// Builder: set search timeout.
    pub fn with_search_timeout(mut self, timeout: Duration) -> Self {
        self.search_timeout = timeout;
        self
    }

    /// Builder: set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Creates configuration from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("REM_INTERVAL_MINUTES") {
            if let Ok(minutes) = val.parse::<u64>() {
                config.interval = Duration::from_secs(minutes * 60);
            }
        }

        if let Ok(val) = std::env::var("REM_MAX_NODES") {
            if let Ok(max) = val.parse::<usize>() {
                config.max_nodes_per_run = max;
            }
        }

        if let Ok(val) = std::env::var("REM_WEB_SEARCH") {
            config.enable_web_search = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = std::env::var("REM_CLUSTERING") {
            config.enable_clustering = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = std::env::var("REM_AUTO_RUN") {
            config.auto_run = val.to_lowercase() == "true" || val == "1";
        }

        config
    }
}

/// Configuration for web search providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Provider name (e.g., "tavily", "searxng", "mock").
    pub provider: String,

    /// API key for the provider (if required).
    pub api_key: Option<String>,

    /// Base URL for the search API.
    pub base_url: Option<String>,

    /// Request timeout.
    pub timeout: Duration,

    /// Maximum results per query.
    pub max_results: usize,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            provider: "searxng".to_string(),
            api_key: None,
            base_url: Some("http://localhost:8080".to_string()),
            timeout: Duration::from_secs(30),
            max_results: 5,
        }
    }
}

impl WebSearchConfig {
    /// Creates a new configuration.
    pub fn new(provider: &str) -> Self {
        Self {
            provider: provider.to_string(),
            ..Default::default()
        }
    }

    /// Builder: set API key.
    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Builder: set base URL.
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = Some(url);
        self
    }

    /// Builder: set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder: set max results.
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max.max(1);
        self
    }

    /// Creates configuration from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("WEB_SEARCH_PROVIDER") {
            config.provider = val;
        }

        if let Ok(val) = std::env::var("WEB_SEARCH_API_KEY") {
            config.api_key = Some(val);
        }

        if let Ok(val) = std::env::var("WEB_SEARCH_BASE_URL") {
            config.base_url = Some(val);
        }

        if let Ok(val) = std::env::var("WEB_SEARCH_MAX_RESULTS") {
            if let Ok(max) = val.parse::<usize>() {
                config.max_results = max;
            }
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rem_phase_config_default() {
        let config = RemPhaseConfig::default();
        assert_eq!(config.interval, Duration::from_secs(3600));
        assert_eq!(config.max_nodes_per_run, 100);
        assert!(config.enable_web_search);
        assert!(config.enable_clustering);
        assert!(!config.auto_run);
    }

    #[test]
    fn test_rem_phase_config_builder() {
        let config = RemPhaseConfig::new()
            .with_interval(Duration::from_secs(1800))
            .with_max_nodes(50)
            .with_web_search(false)
            .with_clustering(false)
            .with_auto_run(true)
            .with_batch_size(20);

        assert_eq!(config.interval, Duration::from_secs(1800));
        assert_eq!(config.max_nodes_per_run, 50);
        assert!(!config.enable_web_search);
        assert!(!config.enable_clustering);
        assert!(config.auto_run);
        assert_eq!(config.batch_size, 20);
    }

    #[test]
    fn test_web_search_config_default() {
        let config = WebSearchConfig::default();
        assert_eq!(config.provider, "searxng");
        assert!(config.api_key.is_none());
        assert_eq!(config.base_url, Some("http://localhost:8080".to_string()));
        assert_eq!(config.max_results, 5);
    }

    #[test]
    fn test_web_search_config_builder() {
        let config = WebSearchConfig::new("tavily")
            .with_api_key("test-key".to_string())
            .with_base_url("https://api.tavily.com".to_string())
            .with_max_results(10);

        assert_eq!(config.provider, "tavily");
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(
            config.base_url,
            Some("https://api.tavily.com".to_string())
        );
        assert_eq!(config.max_results, 10);
    }

    #[test]
    fn test_cross_namespace_threshold_clamping() {
        let config = RemPhaseConfig::new().with_cross_namespace_threshold(1.5);
        assert!((config.cross_namespace_threshold - 1.0).abs() < f32::EPSILON);

        let config = RemPhaseConfig::new().with_cross_namespace_threshold(-0.5);
        assert!((config.cross_namespace_threshold - 0.0).abs() < f32::EPSILON);
    }
}
