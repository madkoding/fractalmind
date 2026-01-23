//! Business services for Fractal-Mind.
//!
//! This module provides:
//! - **Ingestion**: Document extraction and chunking
//! - **REM Phase**: Async learning and memory consolidation
//! - **Web Search**: Provider trait for external information retrieval
//! - **Configuration**: Service-level settings
//!
//! # Ingestion
//!
//! Ingest documents (text, PDF, images) into the knowledge graph:
//!
//! ```ignore
//! use fractalmind::services::ingestion::{IngestionService, IngestionInput};
//!
//! let service = IngestionService::with_defaults();
//! let input = IngestionInput::new(data, "global")
//!     .with_filename("doc.txt");
//!
//! let result = service.ingest(input, embedding_fn).await?;
//! ```
//!
//! # REM Phase
//!
//! The REM phase mimics human sleep cycles for memory consolidation:
//!
//! ```ignore
//! use fractalmind::services::{RemPhaseService, RemPhaseConfig};
//!
//! let config = RemPhaseConfig::new()
//!     .with_web_search(true)
//!     .with_clustering(true);
//!
//! let service = RemPhaseService::new(config, search_provider);
//! let result = service.run_once(incomplete_nodes, embedding_fn).await;
//! ```

#![allow(dead_code)]

pub mod config;
pub mod ingestion;
pub mod rem_phase;
pub mod rem_scheduler;
pub mod storage;
pub mod upload;
pub mod web_search;
pub mod model_conversion;
pub mod fractal_builder;

// Re-exports
pub use config::{RemPhaseConfig, WebSearchConfig};
pub use fractal_builder::{FractalBuilder, FractalBuilderConfig, FractalBuildResult};
pub use rem_phase::{
    RemPhaseResult, RemPhaseService, RemPhaseServiceBuilder, RemPhaseStatus, SearchStats,
};
pub use rem_scheduler::{RemScheduler, RemSchedulerConfig, RemSchedulerStatus, RemRunResult};
pub use web_search::{
    MockSearchProvider, SearchResponse, SearchResult, WebSearchFactory, WebSearchProvider,
};

// Ingestion re-exports
pub use ingestion::{
    ChunkingResult, FileType, IngestionConfig, IngestionInput, IngestionResult, IngestionService,
    IngestionServiceBuilder, TextChunk, TextChunker,
};

// Model conversion re-exports
pub use model_conversion::ModelConversionService;

// Storage and upload re-exports
pub use storage::StorageManager;
pub use upload::{UploadSessionManager, UploadConfig, UploadCleanupJob, ChunkResult, FinalizeResult};

