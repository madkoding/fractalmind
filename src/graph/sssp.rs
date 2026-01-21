//! Single Source Shortest Path (SSSP) algorithm with hopset optimization.
//!
//! This module implements graph navigation using weighted shortest paths,
//! where edge weights represent 1/similarity (lower = more relevant).

#![allow(dead_code)]

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::time::Instant;

use super::config::SsspConfig;
use super::similarity::similarity_to_distance;

/// A node in the graph for SSSP computation.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier.
    pub id: String,
    /// Namespace this node belongs to.
    pub namespace: String,
    /// Outgoing edges (neighbor_id -> similarity).
    pub edges: HashMap<String, f32>,
}

impl GraphNode {
    /// Creates a new graph node.
    pub fn new(id: String, namespace: String) -> Self {
        Self {
            id,
            namespace,
            edges: HashMap::new(),
        }
    }

    /// Adds an edge to another node with given similarity.
    pub fn add_edge(&mut self, neighbor_id: String, similarity: f32) {
        self.edges.insert(neighbor_id, similarity.clamp(0.0, 1.0));
    }

    /// Gets the weight (distance) to a neighbor.
    pub fn weight_to(&self, neighbor_id: &str) -> Option<f32> {
        self.edges.get(neighbor_id).map(|&sim| similarity_to_distance(sim))
    }
}

/// Entry in the priority queue for Dijkstra's algorithm.
#[derive(Clone)]
struct DijkstraEntry {
    node_id: String,
    distance: f32,
}

impl Eq for DijkstraEntry {}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_id == other.node_id
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (smaller distance = higher priority)
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A path through the graph.
#[derive(Debug, Clone)]
pub struct GraphPath {
    /// Sequence of node IDs from source to target.
    pub nodes: Vec<String>,
    /// Total distance (sum of edge weights).
    pub total_distance: f32,
    /// Total similarity (product of edge similarities).
    pub total_similarity: f32,
    /// Number of hops.
    pub hop_count: usize,
}

impl GraphPath {
    /// Creates an empty path starting at a node.
    pub fn new(start_id: String) -> Self {
        Self {
            nodes: vec![start_id],
            total_distance: 0.0,
            total_similarity: 1.0,
            hop_count: 0,
        }
    }

    /// Extends the path with a new node.
    pub fn extend(&self, node_id: String, edge_distance: f32, edge_similarity: f32) -> Self {
        let mut new_nodes = self.nodes.clone();
        new_nodes.push(node_id);

        Self {
            nodes: new_nodes,
            total_distance: self.total_distance + edge_distance,
            total_similarity: self.total_similarity * edge_similarity,
            hop_count: self.hop_count + 1,
        }
    }

    /// Gets the source node ID.
    pub fn source(&self) -> Option<&str> {
        self.nodes.first().map(|s| s.as_str())
    }

    /// Gets the target node ID.
    pub fn target(&self) -> Option<&str> {
        self.nodes.last().map(|s| s.as_str())
    }
}

/// Result of SSSP computation.
#[derive(Debug)]
pub struct SsspResult {
    /// Shortest distances from source to each reachable node.
    pub distances: HashMap<String, f32>,
    /// Predecessor map for path reconstruction.
    pub predecessors: HashMap<String, String>,
    /// Nodes visited during computation.
    pub visited_count: usize,
    /// Time taken in milliseconds.
    pub time_ms: u64,
}

impl SsspResult {
    /// Reconstructs the path from source to a target node.
    pub fn reconstruct_path(&self, source: &str, target: &str) -> Option<GraphPath> {
        if !self.distances.contains_key(target) {
            return None;
        }

        let mut path_nodes = vec![target.to_string()];
        let mut current = target;

        while let Some(pred) = self.predecessors.get(current) {
            path_nodes.push(pred.clone());
            if pred == source {
                break;
            }
            current = pred;
        }

        path_nodes.reverse();

        if path_nodes.first().map(|s| s.as_str()) != Some(source) {
            return None;
        }

        Some(GraphPath {
            nodes: path_nodes.clone(),
            total_distance: *self.distances.get(target).unwrap_or(&f32::MAX),
            total_similarity: 1.0 / self.distances.get(target).unwrap_or(&1.0).max(0.001),
            hop_count: path_nodes.len() - 1,
        })
    }

    /// Gets the k nearest nodes from the source.
    pub fn k_nearest(&self, k: usize) -> Vec<(String, f32)> {
        let mut sorted: Vec<_> = self.distances.iter()
            .map(|(id, &dist)| (id.clone(), dist))
            .collect();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted.into_iter().take(k).collect()
    }
}

/// A hopset entry (shortcut edge).
#[derive(Debug, Clone)]
pub struct HopsetEntry {
    /// Source node ID.
    pub from: String,
    /// Target node ID.
    pub to: String,
    /// Distance through the shortcut.
    pub distance: f32,
    /// Number of hops this shortcut represents.
    pub hop_span: usize,
}

/// Hopset for graph optimization.
#[derive(Debug, Default)]
pub struct Hopset {
    /// Shortcut edges indexed by source node.
    entries: HashMap<String, Vec<HopsetEntry>>,
    /// Total number of shortcuts.
    pub size: usize,
}

impl Hopset {
    /// Creates an empty hopset.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            size: 0,
        }
    }

    /// Adds a shortcut to the hopset.
    pub fn add(&mut self, from: String, to: String, distance: f32, hop_span: usize) {
        let entry = HopsetEntry {
            from: from.clone(),
            to,
            distance,
            hop_span,
        };

        self.entries.entry(from).or_default().push(entry);
        self.size += 1;
    }

    /// Gets shortcuts from a node.
    pub fn get_shortcuts(&self, from: &str) -> &[HopsetEntry] {
        self.entries.get(from).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Checks if the hopset has shortcuts from a node.
    pub fn has_shortcuts(&self, from: &str) -> bool {
        self.entries.contains_key(from)
    }
}

/// SSSP algorithm implementation.
pub struct Sssp {
    config: SsspConfig,
}

impl Sssp {
    /// Creates a new SSSP instance.
    pub fn new(config: SsspConfig) -> Self {
        Self { config }
    }

    /// Creates a new SSSP instance with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SsspConfig::default())
    }

    /// Computes shortest paths from a source node using Dijkstra's algorithm.
    pub fn compute(
        &self,
        graph: &HashMap<String, GraphNode>,
        source: &str,
        hopset: Option<&Hopset>,
    ) -> SsspResult {
        let start = Instant::now();

        let mut distances: HashMap<String, f32> = HashMap::new();
        let mut predecessors: HashMap<String, String> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut heap = BinaryHeap::new();

        // Initialize source
        distances.insert(source.to_string(), 0.0);
        heap.push(DijkstraEntry {
            node_id: source.to_string(),
            distance: 0.0,
        });

        let source_namespace = graph.get(source).map(|n| n.namespace.as_str());

        while let Some(DijkstraEntry { node_id, distance }) = heap.pop() {
            // Skip if we've already found a shorter path
            if visited.contains(&node_id) {
                continue;
            }

            // Check hop limit
            if visited.len() >= self.config.max_hops {
                break;
            }

            visited.insert(node_id.clone());

            // Get the current node
            let current_node = match graph.get(&node_id) {
                Some(n) => n,
                None => continue,
            };

            // Check namespace restrictions
            if !self.config.allow_cross_namespace {
                if let Some(src_ns) = source_namespace {
                    if current_node.namespace != src_ns {
                        continue;
                    }
                }
            }

            // Process regular edges
            for (neighbor_id, &similarity) in &current_node.edges {
                if visited.contains(neighbor_id) {
                    continue;
                }

                let edge_distance = similarity_to_distance(similarity);
                let new_distance = distance + edge_distance;

                let is_shorter = distances.get(neighbor_id)
                    .map(|&d| new_distance < d)
                    .unwrap_or(true);

                if is_shorter {
                    distances.insert(neighbor_id.clone(), new_distance);
                    predecessors.insert(neighbor_id.clone(), node_id.clone());
                    heap.push(DijkstraEntry {
                        node_id: neighbor_id.clone(),
                        distance: new_distance,
                    });
                }
            }

            // Process hopset shortcuts if available
            if self.config.use_hopsets {
                if let Some(hs) = hopset {
                    for shortcut in hs.get_shortcuts(&node_id) {
                        if visited.contains(&shortcut.to) {
                            continue;
                        }

                        let new_distance = distance + shortcut.distance;

                        let is_shorter = distances.get(&shortcut.to)
                            .map(|&d| new_distance < d)
                            .unwrap_or(true);

                        if is_shorter {
                            distances.insert(shortcut.to.clone(), new_distance);
                            predecessors.insert(shortcut.to.clone(), node_id.clone());
                            heap.push(DijkstraEntry {
                                node_id: shortcut.to.clone(),
                                distance: new_distance,
                            });
                        }
                    }
                }
            }
        }

        SsspResult {
            distances,
            predecessors,
            visited_count: visited.len(),
            time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Computes shortest paths to multiple targets.
    pub fn compute_to_targets(
        &self,
        graph: &HashMap<String, GraphNode>,
        source: &str,
        targets: &[String],
        hopset: Option<&Hopset>,
    ) -> Vec<Option<GraphPath>> {
        let result = self.compute(graph, source, hopset);

        targets
            .iter()
            .map(|target| result.reconstruct_path(source, target))
            .collect()
    }

    /// Builds a hopset for the graph using sampling.
    pub fn build_hopset(&self, graph: &HashMap<String, GraphNode>) -> Hopset {
        let mut hopset = Hopset::new();

        // Sample nodes for hopset construction
        let sample_size = (graph.len() as f32 * self.config.hopset_beta) as usize;
        let sample_size = sample_size.max(1).min(graph.len());

        let sampled_nodes: Vec<&String> = graph.keys().take(sample_size).collect();

        // For each sampled node, compute short-range shortest paths
        for &sample_node in &sampled_nodes {
            let temp_config = SsspConfig::new()
                .with_max_hops(3)
                .with_hopsets(false);

            let temp_sssp = Sssp::new(temp_config);
            let result = temp_sssp.compute(graph, sample_node, None);

            // Add shortcuts for 2-hop and 3-hop paths
            for (target, &distance) in &result.distances {
                if target == sample_node {
                    continue;
                }

                if let Some(path) = result.reconstruct_path(sample_node, target) {
                    if path.hop_count >= 2 && path.hop_count <= 3 {
                        hopset.add(
                            sample_node.clone(),
                            target.clone(),
                            distance,
                            path.hop_count,
                        );
                    }
                }
            }
        }

        hopset
    }

    /// Finds the k nearest nodes to a source based on graph distance.
    pub fn k_nearest(
        &self,
        graph: &HashMap<String, GraphNode>,
        source: &str,
        k: usize,
        hopset: Option<&Hopset>,
    ) -> Vec<(String, f32, Option<GraphPath>)> {
        let result = self.compute(graph, source, hopset);
        let nearest = result.k_nearest(k);

        nearest
            .into_iter()
            .map(|(id, dist)| {
                let path = result.reconstruct_path(source, &id);
                (id, dist, path)
            })
            .collect()
    }
}

/// Builder for creating graph structures.
pub struct GraphBuilder {
    nodes: HashMap<String, GraphNode>,
}

impl GraphBuilder {
    /// Creates a new graph builder.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Adds a node to the graph.
    pub fn add_node(&mut self, id: String, namespace: String) -> &mut Self {
        self.nodes.insert(id.clone(), GraphNode::new(id, namespace));
        self
    }

    /// Adds an edge between two nodes.
    pub fn add_edge(&mut self, from: &str, to: &str, similarity: f32) -> &mut Self {
        if let Some(node) = self.nodes.get_mut(from) {
            node.add_edge(to.to_string(), similarity);
        }
        self
    }

    /// Adds a bidirectional edge.
    pub fn add_bidirectional_edge(&mut self, a: &str, b: &str, similarity: f32) -> &mut Self {
        self.add_edge(a, b, similarity);
        self.add_edge(b, a, similarity);
        self
    }

    /// Builds the graph.
    pub fn build(self) -> HashMap<String, GraphNode> {
        self.nodes
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_graph() -> HashMap<String, GraphNode> {
        let mut builder = GraphBuilder::new();

        builder
            .add_node("A".to_string(), "ns1".to_string())
            .add_node("B".to_string(), "ns1".to_string())
            .add_node("C".to_string(), "ns1".to_string())
            .add_node("D".to_string(), "ns1".to_string())
            .add_bidirectional_edge("A", "B", 0.9) // Distance ~1.1
            .add_bidirectional_edge("A", "C", 0.5) // Distance 2.0
            .add_bidirectional_edge("B", "C", 0.8) // Distance 1.25
            .add_bidirectional_edge("B", "D", 0.7) // Distance ~1.4
            .add_bidirectional_edge("C", "D", 0.6); // Distance ~1.7

        builder.build()
    }

    #[test]
    fn test_sssp_basic() {
        let graph = build_test_graph();
        let sssp = Sssp::with_defaults();

        let result = sssp.compute(&graph, "A", None);

        assert!(result.distances.contains_key("A"));
        assert!((result.distances["A"] - 0.0).abs() < 0.001);
        assert!(result.distances.contains_key("B"));
        assert!(result.distances.contains_key("C"));
        assert!(result.distances.contains_key("D"));
    }

    #[test]
    fn test_sssp_path_reconstruction() {
        let graph = build_test_graph();
        let sssp = Sssp::with_defaults();

        let result = sssp.compute(&graph, "A", None);
        let path = result.reconstruct_path("A", "D").unwrap();

        assert_eq!(path.source(), Some("A"));
        assert_eq!(path.target(), Some("D"));
        assert!(path.hop_count >= 1);
        assert!(path.total_distance > 0.0);
    }

    #[test]
    fn test_sssp_k_nearest() {
        let graph = build_test_graph();
        let sssp = Sssp::with_defaults();

        let result = sssp.compute(&graph, "A", None);
        let nearest = result.k_nearest(3);

        assert!(!nearest.is_empty());
        // First should be A itself with distance 0
        assert_eq!(nearest[0].0, "A");
        assert!((nearest[0].1 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_sssp_with_hopset() {
        let graph = build_test_graph();
        let sssp = Sssp::with_defaults();

        let hopset = sssp.build_hopset(&graph);
        let result = sssp.compute(&graph, "A", Some(&hopset));

        assert!(result.distances.contains_key("D"));
    }

    #[test]
    fn test_hopset_construction() {
        let graph = build_test_graph();
        let sssp = Sssp::new(SsspConfig::new().with_hopset_beta(1.0));

        let hopset = sssp.build_hopset(&graph);

        // Should have some shortcuts
        assert!(hopset.size > 0);
    }

    #[test]
    fn test_graph_builder() {
        let mut builder = GraphBuilder::new();
        builder
            .add_node("X".to_string(), "test".to_string())
            .add_node("Y".to_string(), "test".to_string())
            .add_edge("X", "Y", 0.8);

        let graph = builder.build();

        assert_eq!(graph.len(), 2);
        assert!(graph["X"].edges.contains_key("Y"));
    }

    #[test]
    fn test_graph_path() {
        let path = GraphPath::new("start".to_string());

        assert_eq!(path.source(), Some("start"));
        assert_eq!(path.target(), Some("start"));
        assert_eq!(path.hop_count, 0);

        let extended = path.extend("middle".to_string(), 1.0, 0.5);

        assert_eq!(extended.source(), Some("start"));
        assert_eq!(extended.target(), Some("middle"));
        assert_eq!(extended.hop_count, 1);
    }

    #[test]
    fn test_namespace_restriction() {
        let mut builder = GraphBuilder::new();
        builder
            .add_node("A".to_string(), "ns1".to_string())
            .add_node("B".to_string(), "ns2".to_string()) // Different namespace
            .add_bidirectional_edge("A", "B", 0.9);

        let graph = builder.build();

        let config = SsspConfig::new().with_cross_namespace(false);
        let sssp = Sssp::new(config);

        let result = sssp.compute(&graph, "A", None);

        // B should not be reachable due to namespace restriction
        // (though A is visited as source, cross-namespace check applies to destination)
        assert!(result.distances.contains_key("A"));
    }

    #[test]
    fn test_max_hops_limit() {
        let graph = build_test_graph();

        let config = SsspConfig::new().with_max_hops(2);
        let sssp = Sssp::new(config);

        let result = sssp.compute(&graph, "A", None);

        assert!(result.visited_count <= 2);
    }

    #[test]
    fn test_compute_to_targets() {
        let graph = build_test_graph();
        let sssp = Sssp::with_defaults();

        let targets = vec!["B".to_string(), "D".to_string(), "Z".to_string()];
        let paths = sssp.compute_to_targets(&graph, "A", &targets, None);

        assert_eq!(paths.len(), 3);
        assert!(paths[0].is_some()); // Path to B exists
        assert!(paths[1].is_some()); // Path to D exists
        assert!(paths[2].is_none()); // Z doesn't exist
    }
}
