//! Document ingestion service.
//!
//! This module provides content extraction and ingestion capabilities:
//! - Text files (.txt, .md, .rst)
//! - PDF documents (requires `pdf` feature)
//! - Images with OCR (requires `ocr` feature)
//! - Automatic text chunking for large documents
//!
//! # Example
//!
//! ```ignore
//! use fractalmind::services::ingestion::{IngestionService, IngestionInput};
//!
//! let service = IngestionService::with_defaults();
//!
//! // Ingest a text file
//! let input = IngestionInput::new(data, "global")
//!     .with_filename("document.txt")
//!     .with_source("/path/to/file");
//!
//! let result = service.ingest(input, embedding_fn).await?;
//! println!("Created {} nodes", result.node_count());
//! ```
//!
//! # Features
//!
//! - `pdf` - Enable PDF document extraction
//! - `ocr` - Enable image OCR using Tesseract

#![allow(dead_code)]

pub mod chunker;
pub mod config;
pub mod extractors;
pub mod service;

// Re-exports
pub use chunker::{ChunkingResult, TextChunk, TextChunker};
pub use config::{ConfigError, FileType, IngestionConfig};
pub use extractors::{
    ContentExtractor, ExtractionMetadata, ExtractionResult, ExtractorFactory, TextExtractor,
};
pub use service::{IngestionInput, IngestionResult, IngestionService, IngestionServiceBuilder};

#[cfg(feature = "pdf")]
pub use extractors::PdfExtractor;

#[cfg(feature = "ocr")]
pub use extractors::ImageExtractor;
