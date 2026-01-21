//! Graph algorithms for fractal memory navigation.
//!
//! This module provides:
//! - **RAPTOR**: Recursive Abstractive Processing for Tree-Organized Retrieval
//! - **SSSP**: Single Source Shortest Path with hopset optimization
//! - **Clustering**: Semantic clustering of nodes
//!
//! # Example
//!
//! ```ignore
//! use fractalmind::graph::{Raptor, RaptorConfig, RaptorNode};
//!
//! let config = RaptorConfig::new()
//!     .with_similarity_threshold(0.6)
//!     .with_max_cluster_size(8);
//!
//! let raptor = Raptor::new(config);
//! let tree = raptor.build_tree(nodes);
//! ```

#![allow(dead_code)]

pub mod cluster;
pub mod config;
pub mod raptor;
pub mod similarity;
pub mod sssp;

// Re-exports
pub use cluster::{Cluster, ClusteringResult};
pub use config::{ConfigError, RaptorConfig, SsspConfig};
pub use raptor::{Raptor, RaptorNode, RaptorStats, RaptorTree, RaptorTreeNode};
pub use similarity::{
    average_pairwise_similarity, compute_centroid, cosine_similarity, distance_to_similarity,
    find_k_most_similar, find_most_similar, similarity_matrix, similarity_to_distance,
};
pub use sssp::{GraphBuilder, GraphNode, GraphPath, Hopset, HopsetEntry, Sssp, SsspResult};
