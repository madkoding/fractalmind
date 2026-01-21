//! Graph algorithm configuration.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Configuration for RAPTOR clustering algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorConfig {
    /// Minimum number of nodes to form a cluster.
    pub min_cluster_size: usize,

    /// Maximum number of nodes in a cluster.
    pub max_cluster_size: usize,

    /// Minimum similarity threshold for clustering (0.0-1.0).
    pub similarity_threshold: f32,

    /// Maximum tree depth (0 = unlimited).
    pub max_depth: usize,

    /// Whether to generate summaries for parent nodes.
    pub generate_summaries: bool,

    /// Batch size for processing nodes.
    pub batch_size: usize,
}

impl Default for RaptorConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 2,
            max_cluster_size: 10,
            similarity_threshold: 0.5,
            max_depth: 5,
            generate_summaries: true,
            batch_size: 100,
        }
    }
}

impl RaptorConfig {
    /// Creates a new configuration with custom values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set minimum cluster size.
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size.max(2);
        self
    }

    /// Builder: set maximum cluster size.
    pub fn with_max_cluster_size(mut self, size: usize) -> Self {
        self.max_cluster_size = size.max(self.min_cluster_size);
        self
    }

    /// Builder: set similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Builder: set maximum depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Builder: enable/disable summary generation.
    pub fn with_summaries(mut self, enabled: bool) -> Self {
        self.generate_summaries = enabled;
        self
    }

    /// Builder: set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.min_cluster_size < 2 {
            return Err(ConfigError::InvalidMinClusterSize);
        }
        if self.max_cluster_size < self.min_cluster_size {
            return Err(ConfigError::InvalidMaxClusterSize);
        }
        if self.similarity_threshold < 0.0 || self.similarity_threshold > 1.0 {
            return Err(ConfigError::InvalidSimilarityThreshold);
        }
        if self.batch_size == 0 {
            return Err(ConfigError::InvalidBatchSize);
        }
        Ok(())
    }
}

/// Configuration for SSSP algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsspConfig {
    /// Maximum number of hops to consider.
    pub max_hops: usize,

    /// Whether to use hopsets for optimization.
    pub use_hopsets: bool,

    /// Hopset parameter (beta) for O(m * log^(2/3) n) complexity.
    pub hopset_beta: f32,

    /// Maximum number of paths to return.
    pub max_paths: usize,

    /// Whether to include cross-namespace paths.
    pub allow_cross_namespace: bool,
}

impl Default for SsspConfig {
    fn default() -> Self {
        Self {
            max_hops: 10,
            use_hopsets: true,
            hopset_beta: 0.5,
            max_paths: 5,
            allow_cross_namespace: true,
        }
    }
}

impl SsspConfig {
    /// Creates a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set maximum hops.
    pub fn with_max_hops(mut self, hops: usize) -> Self {
        self.max_hops = hops.max(1);
        self
    }

    /// Builder: enable/disable hopsets.
    pub fn with_hopsets(mut self, enabled: bool) -> Self {
        self.use_hopsets = enabled;
        self
    }

    /// Builder: set hopset beta parameter.
    pub fn with_hopset_beta(mut self, beta: f32) -> Self {
        self.hopset_beta = beta.clamp(0.1, 1.0);
        self
    }

    /// Builder: set maximum paths.
    pub fn with_max_paths(mut self, paths: usize) -> Self {
        self.max_paths = paths.max(1);
        self
    }

    /// Builder: allow/disallow cross-namespace paths.
    pub fn with_cross_namespace(mut self, allowed: bool) -> Self {
        self.allow_cross_namespace = allowed;
        self
    }
}

/// Configuration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    InvalidMinClusterSize,
    InvalidMaxClusterSize,
    InvalidSimilarityThreshold,
    InvalidBatchSize,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMinClusterSize => write!(f, "Minimum cluster size must be at least 2"),
            Self::InvalidMaxClusterSize => {
                write!(f, "Maximum cluster size must be >= minimum cluster size")
            }
            Self::InvalidSimilarityThreshold => {
                write!(f, "Similarity threshold must be between 0.0 and 1.0")
            }
            Self::InvalidBatchSize => write!(f, "Batch size must be at least 1"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raptor_config_default() {
        let config = RaptorConfig::default();
        assert_eq!(config.min_cluster_size, 2);
        assert_eq!(config.max_cluster_size, 10);
        assert!((config.similarity_threshold - 0.5).abs() < f32::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_raptor_config_builder() {
        let config = RaptorConfig::new()
            .with_min_cluster_size(3)
            .with_max_cluster_size(8)
            .with_similarity_threshold(0.7)
            .with_max_depth(3)
            .with_summaries(false)
            .with_batch_size(50);

        assert_eq!(config.min_cluster_size, 3);
        assert_eq!(config.max_cluster_size, 8);
        assert!((config.similarity_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.max_depth, 3);
        assert!(!config.generate_summaries);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_raptor_config_validation() {
        let config = RaptorConfig::new().with_min_cluster_size(1);
        assert_eq!(config.min_cluster_size, 2); // Clamped to minimum

        let config = RaptorConfig::new()
            .with_min_cluster_size(5)
            .with_max_cluster_size(3);
        assert!(config.max_cluster_size >= config.min_cluster_size);
    }

    #[test]
    fn test_sssp_config_default() {
        let config = SsspConfig::default();
        assert_eq!(config.max_hops, 10);
        assert!(config.use_hopsets);
        assert!(config.allow_cross_namespace);
    }

    #[test]
    fn test_sssp_config_builder() {
        let config = SsspConfig::new()
            .with_max_hops(5)
            .with_hopsets(false)
            .with_max_paths(3)
            .with_cross_namespace(false);

        assert_eq!(config.max_hops, 5);
        assert!(!config.use_hopsets);
        assert_eq!(config.max_paths, 3);
        assert!(!config.allow_cross_namespace);
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::InvalidMinClusterSize;
        assert!(err.to_string().contains("Minimum cluster size"));
    }
}
