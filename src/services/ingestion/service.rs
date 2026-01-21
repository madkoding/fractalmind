//! Main ingestion service.
//!
//! Orchestrates the entire ingestion pipeline:
//! 1. File type detection
//! 2. Content extraction
//! 3. Text chunking
//! 4. Node creation

#![allow(dead_code)]

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use tracing::{debug, info};

use crate::models::{EmbeddingVector, FractalNode, NodeMetadata};

use super::chunker::{ChunkingResult, TextChunk, TextChunker};
use super::config::{FileType, IngestionConfig};
use super::extractors::{ExtractionResult, ExtractorFactory};

/// Result of an ingestion operation.
#[derive(Debug)]
pub struct IngestionResult {
    /// Generated nodes (one per chunk).
    pub nodes: Vec<FractalNode>,

    /// Original extraction result.
    pub extraction: ExtractionResult,

    /// Chunking result.
    pub chunking: ChunkingResult,

    /// Total time taken in milliseconds.
    pub time_ms: u64,

    /// Any warnings during processing.
    pub warnings: Vec<String>,
}

impl IngestionResult {
    /// Gets the number of nodes created.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Checks if the ingestion was successful.
    pub fn is_successful(&self) -> bool {
        !self.nodes.is_empty()
    }

    /// Checks if there were warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty() || self.extraction.has_warnings()
    }

    /// Gets all warnings (from extraction and processing).
    pub fn all_warnings(&self) -> Vec<String> {
        let mut all = self.warnings.clone();
        all.extend(self.extraction.warnings.clone());
        all
    }
}

/// Input for ingestion operation.
#[derive(Debug, Clone)]
pub struct IngestionInput {
    /// Raw file content.
    pub data: Vec<u8>,

    /// File name (optional, for type detection).
    pub filename: Option<String>,

    /// Source identifier (e.g., URL, file path).
    pub source: Option<String>,

    /// Namespace to store nodes in.
    pub namespace: String,

    /// Tags to apply to generated nodes.
    pub tags: Vec<String>,

    /// Language hint for OCR.
    pub language: Option<String>,
}

impl IngestionInput {
    /// Creates a new ingestion input.
    pub fn new(data: Vec<u8>, namespace: &str) -> Self {
        Self {
            data,
            filename: None,
            source: None,
            namespace: namespace.to_string(),
            tags: Vec::new(),
            language: None,
        }
    }

    /// Sets the filename.
    pub fn with_filename(mut self, filename: &str) -> Self {
        self.filename = Some(filename.to_string());
        self
    }

    /// Sets the source.
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Sets the tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Sets the language.
    pub fn with_language(mut self, language: &str) -> Self {
        self.language = Some(language.to_string());
        self
    }

    /// Detects the file type from filename extension.
    pub fn detect_file_type(&self) -> FileType {
        if let Some(filename) = &self.filename {
            if let Some(ext) = Path::new(filename).extension() {
                return FileType::from_extension(&ext.to_string_lossy());
            }
        }
        FileType::Unknown
    }
}

/// The main ingestion service.
pub struct IngestionService {
    config: IngestionConfig,
    chunker: TextChunker,
}

impl IngestionService {
    /// Creates a new ingestion service with the given configuration.
    pub fn new(config: IngestionConfig) -> Self {
        let chunker = TextChunker::from_config(&config);
        Self { config, chunker }
    }

    /// Creates a service with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(IngestionConfig::default())
    }

    /// Gets the configuration.
    pub fn config(&self) -> &IngestionConfig {
        &self.config
    }

    /// Ingests content and generates nodes.
    ///
    /// The embedding generator function is called for each chunk to generate embeddings.
    pub async fn ingest<F>(
        &self,
        input: IngestionInput,
        embedding_generator: F,
    ) -> Result<IngestionResult>
    where
        F: Fn(&str) -> EmbeddingVector,
    {
        let start = Instant::now();
        let mut warnings = Vec::new();

        // Validate input
        if input.data.is_empty() {
            return Err(anyhow!("Empty input data"));
        }

        if !self.config.is_size_allowed(input.data.len()) {
            return Err(anyhow!(
                "File size {} exceeds maximum {}",
                input.data.len(),
                self.config.max_file_size
            ));
        }

        // Detect file type
        let file_type = input.detect_file_type();
        debug!("Detected file type: {:?}", file_type);

        if !file_type.is_supported() {
            return Err(anyhow!("Unsupported file type"));
        }

        // Check if feature is enabled for this file type
        if file_type == FileType::Pdf && !self.config.enable_pdf {
            return Err(anyhow!("PDF extraction is disabled"));
        }

        if file_type == FileType::Image && !self.config.enable_ocr {
            return Err(anyhow!("OCR is disabled"));
        }

        // Get extractor
        let extractor = ExtractorFactory::create(file_type)
            .ok_or_else(|| anyhow!("No extractor available for {:?}", file_type))?;

        info!(
            "Using {} for {:?} ({})",
            extractor.name(),
            file_type,
            input.filename.as_deref().unwrap_or("unknown")
        );

        // Extract content
        let extraction = extractor.extract(&input.data).await?;

        if extraction.text.is_empty() {
            warnings.push("No text content extracted".to_string());
        }

        // Chunk the text
        let chunking = self.chunker.chunk(&extraction.text);

        info!(
            "Chunked {} chars into {} chunks (avg {} chars)",
            chunking.original_length, chunking.chunks.len(), chunking.avg_chunk_size
        );

        // Generate nodes
        let nodes = self.create_nodes(&chunking.chunks, &input, &embedding_generator);

        let result = IngestionResult {
            nodes,
            extraction,
            chunking,
            time_ms: start.elapsed().as_millis() as u64,
            warnings,
        };

        info!(
            "Ingestion complete: {} nodes in {}ms",
            result.node_count(),
            result.time_ms
        );

        Ok(result)
    }

    /// Ingests content from a file path.
    pub async fn ingest_file<F>(
        &self,
        path: &Path,
        namespace: &str,
        embedding_generator: F,
    ) -> Result<IngestionResult>
    where
        F: Fn(&str) -> EmbeddingVector,
    {
        let data = tokio::fs::read(path).await?;
        let filename = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string());

        let mut input = IngestionInput::new(data, namespace)
            .with_source(&path.to_string_lossy());

        if let Some(name) = filename {
            input = input.with_filename(&name);
        }

        self.ingest(input, embedding_generator).await
    }

    /// Ingests plain text directly (no extraction needed).
    pub fn ingest_text<F>(
        &self,
        text: &str,
        namespace: &str,
        source: Option<&str>,
        tags: Vec<String>,
        embedding_generator: F,
    ) -> Result<IngestionResult>
    where
        F: Fn(&str) -> EmbeddingVector,
    {
        let start = Instant::now();

        if text.trim().is_empty() {
            return Err(anyhow!("Empty text input"));
        }

        // Create fake extraction result
        let extraction = super::extractors::ExtractionResult::new(text.to_string(), FileType::Text);

        // Chunk the text
        let chunking = self.chunker.chunk(text);

        // Build input for node creation
        let input = IngestionInput {
            data: text.as_bytes().to_vec(),
            filename: None,
            source: source.map(|s| s.to_string()),
            namespace: namespace.to_string(),
            tags,
            language: None,
        };

        // Generate nodes
        let nodes = self.create_nodes(&chunking.chunks, &input, &embedding_generator);

        Ok(IngestionResult {
            nodes,
            extraction,
            chunking,
            time_ms: start.elapsed().as_millis() as u64,
            warnings: Vec::new(),
        })
    }

    /// Creates FractalNode instances from chunks.
    fn create_nodes<F>(
        &self,
        chunks: &[TextChunk],
        input: &IngestionInput,
        embedding_generator: &F,
    ) -> Vec<FractalNode>
    where
        F: Fn(&str) -> EmbeddingVector,
    {
        chunks
            .iter()
            .map(|chunk| {
                // Generate embedding
                let embedding = embedding_generator(&chunk.content);

                // Create metadata
                let mut metadata = NodeMetadata::default();
                metadata.tags = input.tags.clone();

                if let Some(src) = &input.source {
                    metadata.source = src.clone();
                }

                if let Some(lang) = &input.language {
                    metadata.language = lang.clone();
                }

                // Add chunk info to tags
                if chunks.len() > 1 {
                    metadata.tags.push(format!("chunk:{}/{}", chunk.index + 1, chunk.total));
                }

                // Create node
                FractalNode::new_leaf(
                    chunk.content.clone(),
                    embedding,
                    input.namespace.clone(),
                    input.source.clone(),
                    metadata,
                )
            })
            .collect()
    }
}

/// Builder for IngestionService.
pub struct IngestionServiceBuilder {
    config: IngestionConfig,
}

impl IngestionServiceBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            config: IngestionConfig::default(),
        }
    }

    /// Sets the configuration.
    pub fn with_config(mut self, config: IngestionConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Sets the chunk overlap.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.config.chunk_overlap = overlap;
        self
    }

    /// Enables/disables OCR.
    pub fn with_ocr(mut self, enabled: bool) -> Self {
        self.config.enable_ocr = enabled;
        self
    }

    /// Enables/disables PDF extraction.
    pub fn with_pdf(mut self, enabled: bool) -> Self {
        self.config.enable_pdf = enabled;
        self
    }

    /// Builds the service.
    pub fn build(self) -> IngestionService {
        IngestionService::new(self.config)
    }
}

impl Default for IngestionServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EmbeddingModel;

    fn mock_embedding(text: &str) -> EmbeddingVector {
        let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
        let values: Vec<f32> = (0..768)
            .map(|i| ((hash + i as u64) % 100) as f32 / 100.0)
            .collect();
        EmbeddingVector::new(values, EmbeddingModel::NomicEmbedTextV15)
    }

    #[test]
    fn test_ingestion_input() {
        let input = IngestionInput::new(b"test".to_vec(), "global")
            .with_filename("test.txt")
            .with_source("/path/to/file")
            .with_tags(vec!["tag1".to_string()]);

        assert_eq!(input.namespace, "global");
        assert_eq!(input.filename, Some("test.txt".to_string()));
        assert_eq!(input.detect_file_type(), FileType::Text);
    }

    #[test]
    fn test_detect_file_type() {
        let txt = IngestionInput::new(vec![], "ns").with_filename("doc.txt");
        assert_eq!(txt.detect_file_type(), FileType::Text);

        let pdf = IngestionInput::new(vec![], "ns").with_filename("doc.pdf");
        assert_eq!(pdf.detect_file_type(), FileType::Pdf);

        let img = IngestionInput::new(vec![], "ns").with_filename("image.png");
        assert_eq!(img.detect_file_type(), FileType::Image);

        let unknown = IngestionInput::new(vec![], "ns").with_filename("file.xyz");
        assert_eq!(unknown.detect_file_type(), FileType::Unknown);
    }

    #[test]
    fn test_ingest_text() {
        let service = IngestionService::with_defaults();
        let text = "This is a test document with some content.";

        let result = service
            .ingest_text(text, "global", Some("test"), vec![], mock_embedding)
            .unwrap();

        assert!(result.is_successful());
        assert_eq!(result.node_count(), 1);
        assert_eq!(result.extraction.text, text);
    }

    #[test]
    fn test_ingest_large_text() {
        let service = IngestionServiceBuilder::new()
            .with_chunk_size(100)
            .with_chunk_overlap(20)
            .build();

        let text = "Lorem ipsum dolor sit amet. ".repeat(50);

        let result = service
            .ingest_text(&text, "global", None, vec![], mock_embedding)
            .unwrap();

        assert!(result.is_successful());
        assert!(result.node_count() > 1);
        assert!(result.chunking.was_split());
    }

    #[test]
    fn test_ingest_empty_text_fails() {
        let service = IngestionService::with_defaults();
        let result = service.ingest_text("", "global", None, vec![], mock_embedding);

        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_with_tags() {
        let service = IngestionService::with_defaults();
        let tags = vec!["important".to_string(), "technical".to_string()];

        let result = service
            .ingest_text("Test content", "global", None, tags.clone(), mock_embedding)
            .unwrap();

        let node = &result.nodes[0];
        assert!(node.metadata.tags.contains(&"important".to_string()));
        assert!(node.metadata.tags.contains(&"technical".to_string()));
    }

    #[tokio::test]
    async fn test_ingest_bytes() {
        let service = IngestionService::with_defaults();
        let input = IngestionInput::new(b"Hello, world!".to_vec(), "global")
            .with_filename("test.txt");

        let result = service.ingest(input, mock_embedding).await.unwrap();

        assert!(result.is_successful());
        assert_eq!(result.extraction.text, "Hello, world!");
    }

    #[tokio::test]
    async fn test_ingest_empty_fails() {
        let service = IngestionService::with_defaults();
        let input = IngestionInput::new(vec![], "global").with_filename("test.txt");

        let result = service.ingest(input, mock_embedding).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ingest_unknown_type_fails() {
        let service = IngestionService::with_defaults();
        let input = IngestionInput::new(b"data".to_vec(), "global").with_filename("file.xyz");

        let result = service.ingest(input, mock_embedding).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_service_builder() {
        let service = IngestionServiceBuilder::new()
            .with_chunk_size(500)
            .with_chunk_overlap(50)
            .with_ocr(false)
            .with_pdf(false)
            .build();

        assert_eq!(service.config().chunk_size, 500);
        assert_eq!(service.config().chunk_overlap, 50);
        assert!(!service.config().enable_ocr);
        assert!(!service.config().enable_pdf);
    }

    #[test]
    fn test_ingestion_result() {
        let service = IngestionService::with_defaults();
        let result = service
            .ingest_text("Test", "global", None, vec![], mock_embedding)
            .unwrap();

        assert!(result.is_successful());
        assert!(!result.has_warnings());
        assert!(result.time_ms < 1000);
    }
}
