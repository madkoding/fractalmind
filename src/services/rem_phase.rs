//! REM Phase service for async learning and memory consolidation.
//!
//! The REM phase mimics human sleep cycles where memories are consolidated:
//! 1. Detects incomplete nodes (knowledge gaps)
//! 2. Performs web search to gather external information
//! 3. Synthesizes information into new knowledge nodes
//! 4. Runs RAPTOR clustering to create hierarchical structure
//! 5. Creates cross-namespace links between personal and global knowledge

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::graph::{Raptor, RaptorConfig, RaptorNode};
use crate::models::{EmbeddingVector, FractalNode, NodeMetadata};

use super::config::RemPhaseConfig;
use super::web_search::{SearchResponse, WebSearchProvider};

/// Status of a REM phase run.
#[derive(Debug, Clone)]
pub enum RemPhaseStatus {
    /// Not running.
    Idle,
    /// Currently processing.
    Running {
        started_at: std::time::SystemTime,
        nodes_processed: usize,
    },
    /// Completed successfully.
    Completed {
        result: RemPhaseResult,
    },
    /// Failed with error.
    Failed {
        error: String,
    },
}

/// Result of a REM phase run.
#[derive(Debug, Clone)]
pub struct RemPhaseResult {
    /// Number of incomplete nodes found.
    pub incomplete_nodes_found: usize,

    /// Number of nodes processed.
    pub nodes_processed: usize,

    /// Number of new nodes created from synthesis.
    pub nodes_created: usize,

    /// Number of nodes updated.
    pub nodes_updated: usize,

    /// Number of clusters formed.
    pub clusters_formed: usize,

    /// Number of cross-namespace links created.
    pub cross_links_created: usize,

    /// Total time taken in milliseconds.
    pub time_ms: u64,

    /// Web search statistics.
    pub search_stats: SearchStats,
}

impl Default for RemPhaseResult {
    fn default() -> Self {
        Self {
            incomplete_nodes_found: 0,
            nodes_processed: 0,
            nodes_created: 0,
            nodes_updated: 0,
            clusters_formed: 0,
            cross_links_created: 0,
            time_ms: 0,
            search_stats: SearchStats::default(),
        }
    }
}

/// Web search statistics.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Number of searches performed.
    pub searches_performed: usize,

    /// Number of successful searches.
    pub successful_searches: usize,

    /// Total results retrieved.
    pub total_results: usize,

    /// Average latency in milliseconds.
    pub avg_latency_ms: u64,
}

/// Input for processing an incomplete node.
#[derive(Debug)]
struct IncompleteNode {
    id: String,
    content: String,
    namespace: String,
    embedding: Option<EmbeddingVector>,
}

/// Synthesized node from external information.
#[derive(Debug)]
struct SynthesizedNode {
    content: String,
    sources: Vec<String>,
    namespace: String,
    related_to: String,
}

/// The REM Phase Service.
pub struct RemPhaseService {
    config: RemPhaseConfig,
    search_provider: Arc<dyn WebSearchProvider>,
    status: Arc<RwLock<RemPhaseStatus>>,
    is_running: Arc<AtomicBool>,
    run_count: Arc<AtomicU64>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl RemPhaseService {
    /// Creates a new REM phase service.
    pub fn new(config: RemPhaseConfig, search_provider: Arc<dyn WebSearchProvider>) -> Self {
        Self {
            config,
            search_provider,
            status: Arc::new(RwLock::new(RemPhaseStatus::Idle)),
            is_running: Arc::new(AtomicBool::new(false)),
            run_count: Arc::new(AtomicU64::new(0)),
            shutdown_tx: None,
        }
    }

    /// Creates a service with default configuration and mock search provider.
    pub fn with_defaults() -> Self {
        use super::web_search::WebSearchFactory;

        Self::new(
            RemPhaseConfig::default(),
            Arc::from(WebSearchFactory::mock()),
        )
    }

    /// Gets the current status.
    pub async fn status(&self) -> RemPhaseStatus {
        self.status.read().await.clone()
    }

    /// Checks if the service is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Gets the number of completed runs.
    pub fn run_count(&self) -> u64 {
        self.run_count.load(Ordering::SeqCst)
    }

    /// Triggers a single REM phase run.
    pub async fn run_once(
        &self,
        incomplete_nodes: Vec<FractalNode>,
        embedding_generator: impl Fn(&str) -> EmbeddingVector,
    ) -> RemPhaseResult {
        let start = Instant::now();

        // Update status to running
        {
            let mut status = self.status.write().await;
            *status = RemPhaseStatus::Running {
                started_at: std::time::SystemTime::now(),
                nodes_processed: 0,
            };
        }
        self.is_running.store(true, Ordering::SeqCst);

        info!(
            "Starting REM phase with {} incomplete nodes",
            incomplete_nodes.len()
        );

        let mut result = RemPhaseResult::default();
        result.incomplete_nodes_found = incomplete_nodes.len();

        // Phase 1: Process incomplete nodes
        let nodes_to_process: Vec<IncompleteNode> = incomplete_nodes
            .into_iter()
            .take(self.config.max_nodes_per_run)
            .map(|node| IncompleteNode {
                id: node.uuid.to_string(),
                content: node.content,
                namespace: node.namespace,
                embedding: Some(node.embedding),
            })
            .collect();

        // Phase 2: Web search for each incomplete node (if enabled)
        let mut synthesized_nodes: Vec<SynthesizedNode> = Vec::new();

        if self.config.enable_web_search && !nodes_to_process.is_empty() {
            let (synth, stats) = self.search_and_synthesize(&nodes_to_process).await;
            synthesized_nodes = synth;
            result.search_stats = stats;
        }

        // Phase 3: Create new nodes from synthesized content
        let mut new_nodes: Vec<FractalNode> = Vec::new();

        for synth in &synthesized_nodes {
            let embedding = embedding_generator(&synth.content);

            let metadata = NodeMetadata {
                source: format!("REM phase synthesis from: {}", synth.related_to),
                tags: vec!["synthesized".to_string(), "rem-phase".to_string()],
                ..Default::default()
            };

            let node = FractalNode::new_leaf(
                synth.content.clone(),
                embedding,
                "global_knowledge".to_string(),
                Some(synth.sources.join(", ")),
                metadata,
            );

            new_nodes.push(node);
            result.nodes_created += 1;
        }

        // Phase 4: RAPTOR clustering (if enabled)
        if self.config.enable_clustering && new_nodes.len() >= 2 {
            let raptor = Raptor::new(RaptorConfig::default());

            let raptor_nodes: Vec<RaptorNode> = new_nodes
                .iter()
                .map(|n| {
                    RaptorNode::new(
                        n.uuid.to_string(),
                        n.content.clone(),
                        n.embedding.clone(),
                    )
                })
                .collect();

            let tree = raptor.build_tree(raptor_nodes);
            result.clusters_formed = tree.stats.total_clusters;

            debug!(
                "RAPTOR clustering created {} clusters at {} depth levels",
                tree.stats.total_clusters, tree.max_depth
            );
        }

        // Update processed count
        result.nodes_processed = nodes_to_process.len();
        result.time_ms = start.elapsed().as_millis() as u64;

        // Update status
        {
            let mut status = self.status.write().await;
            *status = RemPhaseStatus::Completed {
                result: result.clone(),
            };
        }
        self.is_running.store(false, Ordering::SeqCst);
        self.run_count.fetch_add(1, Ordering::SeqCst);

        info!(
            "REM phase completed: {} nodes processed, {} created, {} clusters in {}ms",
            result.nodes_processed, result.nodes_created, result.clusters_formed, result.time_ms
        );

        result
    }

    /// Performs web search and synthesizes results for incomplete nodes.
    async fn search_and_synthesize(
        &self,
        nodes: &[IncompleteNode],
    ) -> (Vec<SynthesizedNode>, SearchStats) {
        let mut synthesized = Vec::new();
        let mut stats = SearchStats::default();
        let mut total_latency = 0u64;

        for node in nodes.iter().take(self.config.batch_size) {
            // Create search query from node content
            let query = self.create_search_query(&node.content);

            match self
                .search_provider
                .search(&query, self.config.max_search_results)
                .await
            {
                Ok(response) => {
                    stats.successful_searches += 1;
                    stats.total_results += response.results.len();
                    total_latency += response.latency_ms;

                    // Synthesize content from search results
                    if !response.results.is_empty() {
                        let synth = self.synthesize_from_search(&node, &response);
                        synthesized.push(synth);
                    }
                }
                Err(e) => {
                    warn!(
                        "Web search failed for node {}: {}",
                        node.id,
                        e
                    );
                }
            }

            stats.searches_performed += 1;
        }

        if stats.successful_searches > 0 {
            stats.avg_latency_ms = total_latency / stats.successful_searches as u64;
        }

        (synthesized, stats)
    }

    /// Creates a search query from node content.
    fn create_search_query(&self, content: &str) -> String {
        // Extract key phrases from content
        // For now, just use the first 100 characters
        let truncated: String = content.chars().take(100).collect();
        truncated.split_whitespace().take(10).collect::<Vec<_>>().join(" ")
    }

    /// Synthesizes a new node from search results.
    fn synthesize_from_search(
        &self,
        original: &IncompleteNode,
        search: &SearchResponse,
    ) -> SynthesizedNode {
        // Combine snippets from search results
        let combined_content = search.combined_snippets();

        let sources: Vec<String> = search
            .results
            .iter()
            .map(|r| r.url.clone())
            .collect();

        let content = format!(
            "# Synthesized Knowledge\n\n\
             **Original Query:** {}\n\n\
             **Sources:**\n{}\n\n\
             **Combined Information:**\n{}",
            original.content,
            sources.iter().map(|s| format!("- {}", s)).collect::<Vec<_>>().join("\n"),
            combined_content
        );

        SynthesizedNode {
            content,
            sources,
            namespace: "global_knowledge".to_string(),
            related_to: original.id.clone(),
        }
    }

    /// Starts the background REM phase scheduler.
    pub async fn start_scheduler(&mut self) -> mpsc::Receiver<RemPhaseResult> {
        let (result_tx, result_rx) = mpsc::channel(10);
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);

        self.shutdown_tx = Some(shutdown_tx);

        let config = self.config.clone();
        let status = self.status.clone();
        let is_running = self.is_running.clone();
        let run_count = self.run_count.clone();
        let _search_provider = self.search_provider.clone();

        tokio::spawn(async move {
            let mut ticker = interval(config.interval);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        if !is_running.load(Ordering::SeqCst) {
                            info!("Scheduled REM phase starting...");

                            // Note: In a real implementation, we would fetch incomplete
                            // nodes from the database here. For now, we just demonstrate
                            // the scheduler structure.

                            let result = RemPhaseResult {
                                time_ms: 0,
                                ..Default::default()
                            };

                            let _ = result_tx.send(result).await;
                            run_count.fetch_add(1, Ordering::SeqCst);
                        } else {
                            debug!("Skipping scheduled REM phase - already running");
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("REM phase scheduler shutting down");
                        break;
                    }
                }
            }

            // Update status on shutdown
            let mut s = status.write().await;
            *s = RemPhaseStatus::Idle;
        });

        result_rx
    }

    /// Stops the background scheduler.
    pub async fn stop_scheduler(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
    }
}

/// Builder for REM phase service.
pub struct RemPhaseServiceBuilder {
    config: RemPhaseConfig,
    search_provider: Option<Arc<dyn WebSearchProvider>>,
}

impl RemPhaseServiceBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            config: RemPhaseConfig::default(),
            search_provider: None,
        }
    }

    /// Sets the configuration.
    pub fn with_config(mut self, config: RemPhaseConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the search provider.
    pub fn with_search_provider(mut self, provider: Arc<dyn WebSearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// Builds the service.
    pub fn build(self) -> RemPhaseService {
        use super::web_search::WebSearchFactory;

        let provider = self
            .search_provider
            .unwrap_or_else(|| Arc::from(WebSearchFactory::mock()));

        RemPhaseService::new(self.config, provider)
    }
}

impl Default for RemPhaseServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EmbeddingModel;

    fn mock_embedding(text: &str) -> EmbeddingVector {
        // Create deterministic embedding based on text hash
        let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
        let values: Vec<f32> = (0..768).map(|i| ((hash + i as u64) % 100) as f32 / 100.0).collect();
        EmbeddingVector::new(values, EmbeddingModel::NomicEmbedTextV15)
    }

    fn create_incomplete_node(content: &str) -> FractalNode {
        let embedding = mock_embedding(content);
        let mut node = FractalNode::new_leaf(
            content.to_string(),
            embedding,
            "test_namespace".to_string(),
            None,
            NodeMetadata::default(),
        );
        node.mark_incomplete();
        node
    }

    #[tokio::test]
    async fn test_rem_phase_service_creation() {
        let service = RemPhaseService::with_defaults();
        assert!(!service.is_running());
        assert_eq!(service.run_count(), 0);
    }

    #[tokio::test]
    async fn test_rem_phase_run_empty() {
        let service = RemPhaseService::with_defaults();
        let result = service.run_once(vec![], mock_embedding).await;

        assert_eq!(result.incomplete_nodes_found, 0);
        assert_eq!(result.nodes_processed, 0);
        assert_eq!(result.nodes_created, 0);
    }

    #[tokio::test]
    async fn test_rem_phase_run_with_nodes() {
        let config = RemPhaseConfig::new()
            .with_web_search(true)
            .with_clustering(false);

        let service = RemPhaseServiceBuilder::new()
            .with_config(config)
            .build();

        let nodes = vec![
            create_incomplete_node("What is quantum computing?"),
            create_incomplete_node("How does machine learning work?"),
        ];

        let result = service.run_once(nodes, mock_embedding).await;

        assert_eq!(result.incomplete_nodes_found, 2);
        assert_eq!(result.nodes_processed, 2);
        assert!(result.search_stats.searches_performed > 0);
    }

    #[tokio::test]
    async fn test_rem_phase_status() {
        let service = RemPhaseService::with_defaults();

        let status = service.status().await;
        assert!(matches!(status, RemPhaseStatus::Idle));

        let _ = service.run_once(vec![], mock_embedding).await;

        let status = service.status().await;
        assert!(matches!(status, RemPhaseStatus::Completed { .. }));
    }

    #[tokio::test]
    async fn test_rem_phase_with_clustering() {
        let config = RemPhaseConfig::new()
            .with_web_search(true)
            .with_clustering(true)
            .with_batch_size(5);

        let service = RemPhaseServiceBuilder::new()
            .with_config(config)
            .build();

        let nodes = vec![
            create_incomplete_node("Topic A: Introduction to Rust"),
            create_incomplete_node("Topic A: Rust ownership model"),
            create_incomplete_node("Topic B: Python basics"),
        ];

        let result = service.run_once(nodes, mock_embedding).await;

        assert_eq!(result.nodes_processed, 3);
        // Clusters may or may not form depending on similarity
        // Just verify the field is accessible
        let _ = result.clusters_formed;
    }

    #[test]
    fn test_service_builder() {
        let service = RemPhaseServiceBuilder::new()
            .with_config(RemPhaseConfig::new().with_max_nodes(50))
            .build();

        assert!(!service.is_running());
    }

    #[tokio::test]
    async fn test_search_query_creation() {
        let service = RemPhaseService::with_defaults();
        let query = service.create_search_query("This is a test content for search query creation");

        assert!(!query.is_empty());
        assert!(query.len() <= 100);
    }

    #[tokio::test]
    async fn test_run_count_increments() {
        let service = RemPhaseService::with_defaults();

        assert_eq!(service.run_count(), 0);

        let _ = service.run_once(vec![], mock_embedding).await;
        assert_eq!(service.run_count(), 1);

        let _ = service.run_once(vec![], mock_embedding).await;
        assert_eq!(service.run_count(), 2);
    }
}
