//! REM Phase Scheduler - Auto-executes memory consolidation during configured hours
//!
//! Like human sleep, the REM phase runs during "night hours" to consolidate
//! memories without impacting daytime performance.

use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{info, warn, debug, error};
use chrono::{Local, Timelike};

use crate::db::connection::DatabaseConnection;
use crate::db::queries::NodeRepository;
use crate::models::llm::ModelBrain;
use crate::services::FractalBuilder;
use crate::services::fractal_builder::FractalBuilderConfig;

/// Configuration for REM scheduler
#[derive(Debug, Clone)]
pub struct RemSchedulerConfig {
    /// Hour to start REM phase (0-23, default: 2 AM)
    pub start_hour: u32,
    /// Hour to end REM phase (0-23, default: 6 AM)
    pub end_hour: u32,
    /// Interval between REM runs in minutes (default: 30)
    pub interval_minutes: u64,
    /// Maximum nodes to process per run
    pub max_nodes_per_run: usize,
    /// Namespaces to consolidate
    pub namespaces: Vec<String>,
    /// Whether the scheduler is enabled
    pub enabled: bool,
}

impl Default for RemSchedulerConfig {
    fn default() -> Self {
        Self {
            start_hour: 2,   // 2 AM
            end_hour: 6,     // 6 AM
            interval_minutes: 30,
            max_nodes_per_run: 100,
            namespaces: vec!["global_knowledge".to_string()],
            enabled: true,
        }
    }
}

impl RemSchedulerConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let enabled = std::env::var("REM_SCHEDULER_ENABLED")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(true);
        
        let start_hour = std::env::var("REM_START_HOUR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        
        let end_hour = std::env::var("REM_END_HOUR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        
        let interval_minutes = std::env::var("REM_INTERVAL_MINUTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
        
        let max_nodes = std::env::var("REM_MAX_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        
        Self {
            start_hour,
            end_hour,
            interval_minutes,
            max_nodes_per_run: max_nodes,
            namespaces: vec!["global_knowledge".to_string()],
            enabled,
        }
    }
    
    /// Check if current time is within REM window
    pub fn is_rem_time(&self) -> bool {
        let hour = Local::now().hour();
        if self.start_hour < self.end_hour {
            hour >= self.start_hour && hour < self.end_hour
        } else {
            // Handle overnight window (e.g., 22:00 - 06:00)
            hour >= self.start_hour || hour < self.end_hour
        }
    }
}

/// Result of a REM scheduler run
#[derive(Debug, Clone)]
pub struct RemRunResult {
    pub namespace: String,
    pub nodes_processed: usize,
    pub nodes_created: usize,
    pub clusters_formed: usize,
    pub duration_ms: u64,
}

/// REM Phase Scheduler Service
pub struct RemScheduler {
    config: RemSchedulerConfig,
    db: Arc<RwLock<DatabaseConnection>>,
    brain: Arc<RwLock<ModelBrain>>,
    is_running: Arc<RwLock<bool>>,
    last_run: Arc<RwLock<Option<chrono::DateTime<Local>>>>,
}

impl RemScheduler {
    /// Create a new REM scheduler
    pub fn new(
        config: RemSchedulerConfig,
        db: DatabaseConnection,
        brain: ModelBrain,
    ) -> Self {
        Self {
            config,
            db: Arc::new(RwLock::new(db)),
            brain: Arc::new(RwLock::new(brain)),
            is_running: Arc::new(RwLock::new(false)),
            last_run: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Start the background scheduler
    pub fn start(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let scheduler = self.clone();
        
        tokio::spawn(async move {
            info!(
                "REM Scheduler started (active hours: {:02}:00 - {:02}:00, interval: {} min)",
                scheduler.config.start_hour,
                scheduler.config.end_hour,
                scheduler.config.interval_minutes
            );
            
            let mut check_interval = interval(Duration::from_secs(60)); // Check every minute
            
            loop {
                check_interval.tick().await;
                
                if !scheduler.config.enabled {
                    continue;
                }
                
                // Check if it's REM time
                if !scheduler.config.is_rem_time() {
                    debug!("Not in REM window, skipping...");
                    continue;
                }
                
                // Check if enough time has passed since last run
                let should_run = {
                    let last_run = scheduler.last_run.read().await;
                    match *last_run {
                        None => true,
                        Some(last) => {
                            let elapsed = Local::now().signed_duration_since(last);
                            elapsed.num_minutes() >= scheduler.config.interval_minutes as i64
                        }
                    }
                };
                
                if !should_run {
                    continue;
                }
                
                // Check if already running
                {
                    let is_running = scheduler.is_running.read().await;
                    if *is_running {
                        debug!("REM phase already running, skipping...");
                        continue;
                    }
                }
                
                // Run REM phase
                info!("🌙 Starting automatic REM phase consolidation...");
                
                {
                    let mut is_running = scheduler.is_running.write().await;
                    *is_running = true;
                }
                
                for namespace in &scheduler.config.namespaces {
                    match scheduler.run_for_namespace(namespace).await {
                        Ok(result) => {
                            info!(
                                "✅ REM completed for '{}': {} processed, {} created, {} clusters ({}ms)",
                                result.namespace,
                                result.nodes_processed,
                                result.nodes_created,
                                result.clusters_formed,
                                result.duration_ms
                            );
                        }
                        Err(e) => {
                            error!("❌ REM failed for '{}': {}", namespace, e);
                        }
                    }
                }
                
                {
                    let mut is_running = scheduler.is_running.write().await;
                    *is_running = false;
                    let mut last_run = scheduler.last_run.write().await;
                    *last_run = Some(Local::now());
                }
                
                info!("🌙 REM phase consolidation completed");
            }
        })
    }
    
    /// Run REM phase for a specific namespace
    async fn run_for_namespace(&self, namespace: &str) -> Result<RemRunResult, String> {
        let start = std::time::Instant::now();
        
        let db = self.db.read().await;
        let brain = self.brain.read().await;
        let node_repo = NodeRepository::new(&db);
        
        // Get leaf nodes
        let all_nodes = node_repo
            .get_by_namespace(namespace)
            .await
            .map_err(|e| format!("Failed to list nodes: {}", e))?;
        
        let leaf_nodes: Vec<_> = all_nodes
            .into_iter()
            .filter(|n| n.depth_level == 0)
            .take(self.config.max_nodes_per_run)
            .collect();
        
        let nodes_processed = leaf_nodes.len();
        let mut nodes_created = 0;
        let mut clusters_formed = 0;
        
        // Build fractal hierarchy if we have enough nodes
        if leaf_nodes.len() >= 3 {
            let config = FractalBuilderConfig::new()
                .with_summaries(true)
                .with_min_nodes(3);
            
            let fractal_builder = FractalBuilder::new(&db, config);
            
            match fractal_builder.build_for_namespace(namespace, Some(&brain)).await {
                Ok(result) => {
                    nodes_created = result.parent_nodes_created;
                    clusters_formed = result.edges_created;
                }
                Err(e) => {
                    warn!("RAPTOR clustering failed: {}", e);
                }
            }
        }
        
        Ok(RemRunResult {
            namespace: namespace.to_string(),
            nodes_processed,
            nodes_created,
            clusters_formed,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Get scheduler status
    pub async fn status(&self) -> RemSchedulerStatus {
        let is_running = *self.is_running.read().await;
        let last_run = *self.last_run.read().await;
        let is_rem_time = self.config.is_rem_time();
        
        RemSchedulerStatus {
            enabled: self.config.enabled,
            is_running,
            is_rem_time,
            last_run,
            start_hour: self.config.start_hour,
            end_hour: self.config.end_hour,
            interval_minutes: self.config.interval_minutes,
        }
    }
}

/// Status information for the REM scheduler
#[derive(Debug, Clone)]
pub struct RemSchedulerStatus {
    pub enabled: bool,
    pub is_running: bool,
    pub is_rem_time: bool,
    pub last_run: Option<chrono::DateTime<Local>>,
    pub start_hour: u32,
    pub end_hour: u32,
    pub interval_minutes: u64,
}
