//! RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) algorithm.
//!
//! This module implements hierarchical clustering of nodes to build a fractal tree structure.
//! The algorithm groups semantically similar nodes and creates parent summary nodes.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::models::EmbeddingVector;

use super::cluster::{Cluster, ClusteringResult};
use super::config::RaptorConfig;
use super::similarity::{
    average_pairwise_similarity, compute_centroid, cosine_similarity, similarity_matrix,
};

/// Input node for RAPTOR clustering.
#[derive(Debug, Clone)]
pub struct RaptorNode {
    /// Unique identifier.
    pub id: String,
    /// Content text.
    pub content: String,
    /// Embedding vector.
    pub embedding: EmbeddingVector,
}

impl RaptorNode {
    /// Creates a new RAPTOR node.
    pub fn new(id: String, content: String, embedding: EmbeddingVector) -> Self {
        Self {
            id,
            content,
            embedding,
        }
    }
}

/// RAPTOR tree node representing a node in the hierarchy.
#[derive(Debug, Clone)]
pub struct RaptorTreeNode {
    /// Cluster ID.
    pub cluster_id: String,
    /// Member node IDs (for leaf clusters) or child cluster IDs.
    pub members: Vec<String>,
    /// Centroid embedding.
    pub centroid: EmbeddingVector,
    /// Summary text (generated for non-leaf nodes).
    pub summary: Option<String>,
    /// Combined content from members.
    pub combined_content: String,
    /// Tree depth (0 = leaf level).
    pub depth: usize,
    /// Parent cluster ID.
    pub parent_id: Option<String>,
    /// Child cluster IDs.
    pub children: Vec<String>,
    /// Internal similarity score.
    pub internal_similarity: f32,
}

/// Result of RAPTOR tree construction.
#[derive(Debug)]
pub struct RaptorTree {
    /// All tree nodes indexed by cluster ID.
    pub nodes: HashMap<String, RaptorTreeNode>,
    /// Root cluster IDs.
    pub roots: Vec<String>,
    /// Leaf cluster IDs.
    pub leaves: Vec<String>,
    /// Maximum depth of the tree.
    pub max_depth: usize,
    /// Total time taken in milliseconds.
    pub build_time_ms: u64,
    /// Statistics about the tree.
    pub stats: RaptorStats,
}

/// Statistics about the RAPTOR tree.
#[derive(Debug, Default)]
pub struct RaptorStats {
    /// Total number of clusters.
    pub total_clusters: usize,
    /// Number of clusters at each depth level.
    pub clusters_per_depth: Vec<usize>,
    /// Average cluster size.
    pub average_cluster_size: f32,
    /// Average internal similarity.
    pub average_internal_similarity: f32,
}

/// RAPTOR algorithm implementation.
pub struct Raptor {
    config: RaptorConfig,
}

impl Raptor {
    /// Creates a new RAPTOR instance with the given configuration.
    pub fn new(config: RaptorConfig) -> Self {
        Self { config }
    }

    /// Creates a new RAPTOR instance with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RaptorConfig::default())
    }

    /// Builds a RAPTOR tree from a list of nodes.
    pub fn build_tree(&self, nodes: Vec<RaptorNode>) -> RaptorTree {
        let start = Instant::now();

        if nodes.is_empty() {
            return RaptorTree {
                nodes: HashMap::new(),
                roots: vec![],
                leaves: vec![],
                max_depth: 0,
                build_time_ms: start.elapsed().as_millis() as u64,
                stats: RaptorStats::default(),
            };
        }

        let mut tree_nodes: HashMap<String, RaptorTreeNode> = HashMap::new();
        let mut current_level_ids: Vec<String> = Vec::new();

        // Step 1: Create leaf clusters from input nodes
        let leaf_clusters = self.create_initial_clusters(&nodes);

        for cluster in leaf_clusters {
            let members: Vec<&RaptorNode> = cluster
                .members()
                .iter()
                .filter_map(|id| nodes.iter().find(|n| &n.id == id))
                .collect();

            let embeddings: Vec<&EmbeddingVector> = members.iter().map(|n| &n.embedding).collect();
            let centroid = compute_centroid(&embeddings)
                .unwrap_or_else(|| members[0].embedding.clone());

            let combined_content: String = members.iter().map(|n| n.content.as_str()).collect::<Vec<_>>().join("\n\n");

            let internal_sim = if embeddings.len() > 1 {
                average_pairwise_similarity(&embeddings)
            } else {
                1.0
            };

            let tree_node = RaptorTreeNode {
                cluster_id: cluster.id().to_string(),
                members: cluster.members().to_vec(),
                centroid,
                summary: None,
                combined_content,
                depth: 0,
                parent_id: None,
                children: vec![],
                internal_similarity: internal_sim,
            };

            current_level_ids.push(cluster.id().to_string());
            tree_nodes.insert(cluster.id().to_string(), tree_node);
        }

        let leaves = current_level_ids.clone();
        let mut max_depth = 0;

        // Step 2: Recursively build parent levels
        let mut depth = 1;
        while current_level_ids.len() > 1 && (self.config.max_depth == 0 || depth <= self.config.max_depth) {
            let current_nodes: Vec<(&String, &EmbeddingVector)> = current_level_ids
                .iter()
                .filter_map(|id| {
                    tree_nodes.get(id).map(|n| (id, &n.centroid))
                })
                .collect();

            if current_nodes.len() < 2 {
                break;
            }

            // Cluster the current level
            let parent_clusters = self.cluster_by_similarity(&current_nodes);

            let mut next_level_ids = Vec::new();

            for cluster in parent_clusters {
                if cluster.size() < 2 && next_level_ids.len() + current_level_ids.len() > cluster.size() {
                    // Skip singleton clusters at higher levels
                    continue;
                }

                let child_ids = cluster.members().to_vec();
                let child_embeddings: Vec<&EmbeddingVector> = child_ids
                    .iter()
                    .filter_map(|id| tree_nodes.get(id).map(|n| &n.centroid))
                    .collect();

                let centroid = compute_centroid(&child_embeddings)
                    .unwrap_or_else(|| child_embeddings[0].clone());

                let combined_content: String = child_ids
                    .iter()
                    .filter_map(|id| tree_nodes.get(id))
                    .map(|n| {
                        n.summary
                            .as_ref()
                            .map(|s| s.as_str())
                            .unwrap_or(&n.combined_content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n");

                let internal_sim = if child_embeddings.len() > 1 {
                    average_pairwise_similarity(&child_embeddings)
                } else {
                    1.0
                };

                let parent_node = RaptorTreeNode {
                    cluster_id: cluster.id().to_string(),
                    members: vec![], // Parent nodes don't have original members
                    centroid,
                    summary: None, // To be filled by LLM later
                    combined_content,
                    depth,
                    parent_id: None,
                    children: child_ids.clone(),
                    internal_similarity: internal_sim,
                };

                // Update children to point to parent
                for child_id in &child_ids {
                    if let Some(child) = tree_nodes.get_mut(child_id) {
                        child.parent_id = Some(cluster.id().to_string());
                    }
                }

                next_level_ids.push(cluster.id().to_string());
                tree_nodes.insert(cluster.id().to_string(), parent_node);
            }

            if next_level_ids.is_empty() {
                break;
            }

            current_level_ids = next_level_ids;
            max_depth = depth;
            depth += 1;
        }

        // Roots are the nodes at the highest level
        let roots = current_level_ids;

        // Calculate statistics
        let stats = self.calculate_stats(&tree_nodes, max_depth);

        RaptorTree {
            nodes: tree_nodes,
            roots,
            leaves,
            max_depth,
            build_time_ms: start.elapsed().as_millis() as u64,
            stats,
        }
    }

    /// Creates initial clusters from leaf nodes using agglomerative clustering.
    fn create_initial_clusters(&self, nodes: &[RaptorNode]) -> Vec<Cluster> {
        if nodes.is_empty() {
            return vec![];
        }

        let embeddings: Vec<&EmbeddingVector> = nodes.iter().map(|n| &n.embedding).collect();
        let ids: Vec<&String> = nodes.iter().map(|n| &n.id).collect();

        self.agglomerative_cluster(&ids, &embeddings)
    }

    /// Clusters nodes by similarity using agglomerative clustering.
    fn cluster_by_similarity(&self, nodes: &[(&String, &EmbeddingVector)]) -> Vec<Cluster> {
        if nodes.is_empty() {
            return vec![];
        }

        let ids: Vec<&String> = nodes.iter().map(|(id, _)| *id).collect();
        let embeddings: Vec<&EmbeddingVector> = nodes.iter().map(|(_, e)| *e).collect();

        self.agglomerative_cluster(&ids, &embeddings)
    }

    /// Performs agglomerative (bottom-up) clustering.
    fn agglomerative_cluster(
        &self,
        ids: &[&String],
        embeddings: &[&EmbeddingVector],
    ) -> Vec<Cluster> {
        if ids.is_empty() {
            return vec![];
        }

        let n = ids.len();

        // Each node starts as its own cluster
        let mut clusters: Vec<Cluster> = ids
            .iter()
            .map(|id| Cluster::new(vec![(*id).clone()]))
            .collect();

        // Compute similarity matrix
        let sim_matrix = similarity_matrix(embeddings);

        // Track which clusters are still active
        let mut active: Vec<bool> = vec![true; n];

        // Map from cluster index to current cluster
        let mut cluster_map: HashMap<usize, usize> = (0..n).map(|i| (i, i)).collect();

        loop {
            // Find the most similar pair of active clusters
            let mut best_sim = f32::NEG_INFINITY;
            let mut best_i = 0;
            let mut best_j = 0;

            for i in 0..n {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !active[j] {
                        continue;
                    }

                    // Average linkage: use precomputed similarity
                    let sim = sim_matrix[i][j];
                    if sim > best_sim {
                        best_sim = sim;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Check if we should stop
            if best_sim < self.config.similarity_threshold {
                break;
            }

            // Get the actual cluster indices
            let cluster_i = cluster_map[&best_i];
            let cluster_j = cluster_map[&best_j];

            // Check if merging would exceed max cluster size
            let merged_size = clusters[cluster_i].size() + clusters[cluster_j].size();
            if merged_size > self.config.max_cluster_size {
                // Mark this pair as non-mergeable by deactivating one
                active[best_j] = false;
                continue;
            }

            // Merge clusters
            let cluster_to_merge = clusters[cluster_j].clone();
            clusters[cluster_i].merge(&cluster_to_merge);

            // Update cluster map: all nodes in cluster_j now point to cluster_i
            for (_, v) in cluster_map.iter_mut() {
                if *v == cluster_j {
                    *v = cluster_i;
                }
            }

            // Deactivate the merged cluster
            active[best_j] = false;

            // Count remaining active clusters
            let active_count = active.iter().filter(|&&a| a).count();
            if active_count <= 1 {
                break;
            }
        }

        // Collect unique active clusters
        let mut seen: HashSet<usize> = HashSet::new();
        let mut result: Vec<Cluster> = Vec::new();

        for i in 0..n {
            if active[i] {
                let cluster_idx = cluster_map[&i];
                if !seen.contains(&cluster_idx) && clusters[cluster_idx].size() >= self.config.min_cluster_size {
                    seen.insert(cluster_idx);
                    result.push(clusters[cluster_idx].clone());
                }
            }
        }

        // Add singleton clusters that didn't get merged (if below min size)
        for i in 0..n {
            let cluster_idx = cluster_map[&i];
            if clusters[cluster_idx].size() < self.config.min_cluster_size && !seen.contains(&cluster_idx) {
                // Create individual cluster for orphans
                let mut cluster = Cluster::new(vec![ids[i].clone()]);
                cluster.set_depth(0);
                result.push(cluster);
                seen.insert(cluster_idx);
            }
        }

        result
    }

    /// Calculates statistics about the tree.
    fn calculate_stats(
        &self,
        nodes: &HashMap<String, RaptorTreeNode>,
        max_depth: usize,
    ) -> RaptorStats {
        let mut clusters_per_depth = vec![0usize; max_depth + 1];
        let mut total_size = 0usize;
        let mut total_similarity = 0.0f32;
        let mut count = 0usize;

        for node in nodes.values() {
            if node.depth <= max_depth {
                clusters_per_depth[node.depth] += 1;
            }

            if node.depth == 0 {
                total_size += node.members.len();
            } else {
                total_size += node.children.len();
            }

            total_similarity += node.internal_similarity;
            count += 1;
        }

        RaptorStats {
            total_clusters: nodes.len(),
            clusters_per_depth,
            average_cluster_size: if count > 0 {
                total_size as f32 / count as f32
            } else {
                0.0
            },
            average_internal_similarity: if count > 0 {
                total_similarity / count as f32
            } else {
                0.0
            },
        }
    }

    /// Performs bottom-up clustering and returns the clustering result.
    pub fn cluster(&self, nodes: &[RaptorNode]) -> ClusteringResult {
        let start = Instant::now();

        let clusters = self.create_initial_clusters(nodes);
        let orphan_count = nodes.len() - clusters.iter().map(|c| c.size()).sum::<usize>();

        ClusteringResult {
            clusters,
            iterations: 1, // Agglomerative is single-pass
            time_ms: start.elapsed().as_millis() as u64,
            orphan_count,
        }
    }
}

impl RaptorTree {
    /// Gets a node by its cluster ID.
    pub fn get_node(&self, cluster_id: &str) -> Option<&RaptorTreeNode> {
        self.nodes.get(cluster_id)
    }

    /// Gets all nodes at a specific depth level.
    pub fn get_nodes_at_depth(&self, depth: usize) -> Vec<&RaptorTreeNode> {
        self.nodes.values().filter(|n| n.depth == depth).collect()
    }

    /// Gets the path from a leaf to the root.
    pub fn get_path_to_root(&self, leaf_id: &str) -> Vec<&RaptorTreeNode> {
        let mut path = Vec::new();
        let mut current_id = Some(leaf_id.to_string());

        while let Some(id) = current_id {
            if let Some(node) = self.nodes.get(&id) {
                path.push(node);
                current_id = node.parent_id.clone();
            } else {
                break;
            }
        }

        path
    }

    /// Finds the most similar leaf cluster to a query embedding.
    pub fn find_similar_leaf(&self, query: &EmbeddingVector) -> Option<(&RaptorTreeNode, f32)> {
        let mut best: Option<(&RaptorTreeNode, f32)> = None;

        for leaf_id in &self.leaves {
            if let Some(node) = self.nodes.get(leaf_id) {
                let sim = cosine_similarity(query, &node.centroid);
                if best.is_none() || sim > best.unwrap().1 {
                    best = Some((node, sim));
                }
            }
        }

        best
    }

    /// Traverses the tree from roots down, collecting relevant nodes.
    pub fn traverse_from_roots(
        &self,
        query: &EmbeddingVector,
        threshold: f32,
    ) -> Vec<&RaptorTreeNode> {
        let mut result = Vec::new();
        let mut to_visit: Vec<&str> = self.roots.iter().map(|s| s.as_str()).collect();

        while let Some(node_id) = to_visit.pop() {
            if let Some(node) = self.nodes.get(node_id) {
                let sim = cosine_similarity(query, &node.centroid);

                if sim >= threshold {
                    result.push(node);

                    // Add children to visit
                    for child_id in &node.children {
                        to_visit.push(child_id);
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EmbeddingModel;

    fn make_node(id: &str, content: &str, values: Vec<f32>) -> RaptorNode {
        RaptorNode::new(
            id.to_string(),
            content.to_string(),
            EmbeddingVector::new(values, EmbeddingModel::NomicEmbedTextV15),
        )
    }

    #[test]
    fn test_raptor_empty_input() {
        let raptor = Raptor::with_defaults();
        let tree = raptor.build_tree(vec![]);

        assert!(tree.nodes.is_empty());
        assert!(tree.roots.is_empty());
        assert!(tree.leaves.is_empty());
    }

    #[test]
    fn test_raptor_single_node() {
        let raptor = Raptor::with_defaults();
        let nodes = vec![make_node("n1", "Content 1", vec![1.0, 0.0, 0.0])];

        let tree = raptor.build_tree(nodes);

        assert_eq!(tree.leaves.len(), 1);
        assert_eq!(tree.roots.len(), 1);
    }

    #[test]
    fn test_raptor_similar_nodes_cluster() {
        let config = RaptorConfig::new()
            .with_similarity_threshold(0.8)
            .with_min_cluster_size(2);

        let raptor = Raptor::new(config);

        // Two similar nodes and one different
        let nodes = vec![
            make_node("n1", "Content 1", vec![1.0, 0.0, 0.0]),
            make_node("n2", "Content 2", vec![0.99, 0.1, 0.0]), // Similar to n1
            make_node("n3", "Content 3", vec![0.0, 1.0, 0.0]),  // Different
        ];

        let tree = raptor.build_tree(nodes);

        // Should have some clustering
        assert!(!tree.leaves.is_empty());
    }

    #[test]
    fn test_raptor_clustering_result() {
        let raptor = Raptor::with_defaults();

        let nodes = vec![
            make_node("n1", "Content 1", vec![1.0, 0.0]),
            make_node("n2", "Content 2", vec![0.9, 0.1]),
            make_node("n3", "Content 3", vec![0.0, 1.0]),
            make_node("n4", "Content 4", vec![0.1, 0.9]),
        ];

        let result = raptor.cluster(&nodes);

        assert!(result.time_ms < 1000); // Should be fast
        assert!(!result.clusters.is_empty());
    }

    #[test]
    fn test_raptor_tree_traversal() {
        let config = RaptorConfig::new()
            .with_similarity_threshold(0.5)
            .with_min_cluster_size(2)
            .with_max_depth(2);

        let raptor = Raptor::new(config);

        let nodes = vec![
            make_node("n1", "Topic A content", vec![1.0, 0.0, 0.0]),
            make_node("n2", "Topic A more", vec![0.95, 0.1, 0.0]),
            make_node("n3", "Topic B content", vec![0.0, 1.0, 0.0]),
            make_node("n4", "Topic B more", vec![0.1, 0.95, 0.0]),
        ];

        let tree = raptor.build_tree(nodes);

        // Query similar to Topic A
        let query = EmbeddingVector::new(vec![0.9, 0.1, 0.0], EmbeddingModel::NomicEmbedTextV15);
        let relevant = tree.traverse_from_roots(&query, 0.3);

        assert!(!relevant.is_empty());
    }

    #[test]
    fn test_raptor_find_similar_leaf() {
        let raptor = Raptor::with_defaults();

        let nodes = vec![
            make_node("n1", "Content 1", vec![1.0, 0.0]),
            make_node("n2", "Content 2", vec![0.0, 1.0]),
        ];

        let tree = raptor.build_tree(nodes);

        let query = EmbeddingVector::new(vec![0.9, 0.1], EmbeddingModel::NomicEmbedTextV15);
        let (leaf, sim) = tree.find_similar_leaf(&query).unwrap();

        assert!(sim > 0.5);
        assert!(!leaf.cluster_id.is_empty());
    }

    #[test]
    fn test_raptor_path_to_root() {
        let config = RaptorConfig::new()
            .with_similarity_threshold(0.3)
            .with_min_cluster_size(2)
            .with_max_depth(3);

        let raptor = Raptor::new(config);

        let nodes = vec![
            make_node("n1", "A", vec![1.0, 0.0]),
            make_node("n2", "B", vec![0.9, 0.1]),
            make_node("n3", "C", vec![0.8, 0.2]),
            make_node("n4", "D", vec![0.7, 0.3]),
        ];

        let tree = raptor.build_tree(nodes);

        if let Some(leaf_id) = tree.leaves.first() {
            let path = tree.get_path_to_root(leaf_id);
            assert!(!path.is_empty());
            // Path should go from leaf to root
            if path.len() > 1 {
                assert!(path[0].depth <= path[path.len() - 1].depth || path[path.len() - 1].parent_id.is_none());
            }
        }
    }

    #[test]
    fn test_raptor_stats() {
        let raptor = Raptor::with_defaults();

        let nodes = vec![
            make_node("n1", "A", vec![1.0, 0.0]),
            make_node("n2", "B", vec![0.9, 0.1]),
            make_node("n3", "C", vec![0.0, 1.0]),
            make_node("n4", "D", vec![0.1, 0.9]),
        ];

        let tree = raptor.build_tree(nodes);

        assert!(tree.stats.total_clusters > 0);
        assert!(tree.stats.average_cluster_size > 0.0);
        assert!(tree.stats.average_internal_similarity >= 0.0);
        assert!(tree.stats.average_internal_similarity <= 1.0);
    }
}
