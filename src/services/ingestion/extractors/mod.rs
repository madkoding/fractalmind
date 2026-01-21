//! Content extractors for different file types.

#![allow(dead_code)]

pub mod text;

#[cfg(feature = "pdf")]
pub mod pdf;

#[cfg(feature = "ocr")]
pub mod image;

use anyhow::Result;
use async_trait::async_trait;

use super::config::FileType;

/// Result of content extraction.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted text content.
    pub text: String,

    /// File type that was processed.
    pub file_type: FileType,

    /// Original file name (if available).
    pub filename: Option<String>,

    /// Metadata extracted from the file.
    pub metadata: ExtractionMetadata,

    /// Any warnings during extraction.
    pub warnings: Vec<String>,
}

impl ExtractionResult {
    /// Creates a new extraction result.
    pub fn new(text: String, file_type: FileType) -> Self {
        Self {
            text,
            file_type,
            filename: None,
            metadata: ExtractionMetadata::default(),
            warnings: Vec::new(),
        }
    }

    /// Sets the filename.
    pub fn with_filename(mut self, filename: String) -> Self {
        self.filename = Some(filename);
        self
    }

    /// Sets the metadata.
    pub fn with_metadata(mut self, metadata: ExtractionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Adds a warning.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Gets the text length.
    pub fn text_length(&self) -> usize {
        self.text.len()
    }

    /// Checks if the extraction was successful (has content).
    pub fn is_successful(&self) -> bool {
        !self.text.is_empty()
    }

    /// Checks if there were warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Metadata extracted from a file.
#[derive(Debug, Clone, Default)]
pub struct ExtractionMetadata {
    /// Document title (if available).
    pub title: Option<String>,

    /// Document author (if available).
    pub author: Option<String>,

    /// Creation date (if available).
    pub created_at: Option<String>,

    /// Number of pages (for PDFs).
    pub page_count: Option<usize>,

    /// Detected language.
    pub language: Option<String>,

    /// File size in bytes.
    pub file_size: Option<usize>,

    /// MIME type.
    pub mime_type: Option<String>,

    /// Character encoding.
    pub encoding: Option<String>,

    /// Custom metadata key-value pairs.
    pub custom: std::collections::HashMap<String, String>,
}

impl ExtractionMetadata {
    /// Creates new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the title.
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Sets the author.
    pub fn with_author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Sets the page count.
    pub fn with_page_count(mut self, count: usize) -> Self {
        self.page_count = Some(count);
        self
    }

    /// Sets the file size.
    pub fn with_file_size(mut self, size: usize) -> Self {
        self.file_size = Some(size);
        self
    }

    /// Sets the MIME type.
    pub fn with_mime_type(mut self, mime: String) -> Self {
        self.mime_type = Some(mime);
        self
    }

    /// Adds a custom metadata field.
    pub fn with_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }
}

/// Trait for content extractors.
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    /// Extracts text content from bytes.
    async fn extract(&self, data: &[u8]) -> Result<ExtractionResult>;

    /// Extracts text content from a file path.
    async fn extract_from_file(&self, path: &std::path::Path) -> Result<ExtractionResult> {
        let data = tokio::fs::read(path).await?;
        let mut result = self.extract(&data).await?;
        result.filename = path.file_name().map(|s| s.to_string_lossy().to_string());
        result.metadata.file_size = Some(data.len());
        Ok(result)
    }

    /// Returns the file types this extractor supports.
    fn supported_types(&self) -> Vec<FileType>;

    /// Checks if this extractor supports a given file type.
    fn supports(&self, file_type: FileType) -> bool {
        self.supported_types().contains(&file_type)
    }

    /// Returns the extractor name.
    fn name(&self) -> &str;
}

/// Factory for creating extractors.
pub struct ExtractorFactory;

impl ExtractorFactory {
    /// Creates an extractor for the given file type.
    pub fn create(file_type: FileType) -> Option<Box<dyn ContentExtractor>> {
        match file_type {
            FileType::Text => Some(Box::new(text::TextExtractor::new())),
            #[cfg(feature = "pdf")]
            FileType::Pdf => Some(Box::new(pdf::PdfExtractor::new())),
            #[cfg(not(feature = "pdf"))]
            FileType::Pdf => None,
            #[cfg(feature = "ocr")]
            FileType::Image => Some(Box::new(image::ImageExtractor::new())),
            #[cfg(not(feature = "ocr"))]
            FileType::Image => None,
            FileType::Unknown => None,
        }
    }

    /// Creates a text extractor.
    pub fn text() -> Box<dyn ContentExtractor> {
        Box::new(text::TextExtractor::new())
    }

    /// Creates a PDF extractor (if feature enabled).
    #[cfg(feature = "pdf")]
    pub fn pdf() -> Box<dyn ContentExtractor> {
        Box::new(pdf::PdfExtractor::new())
    }

    /// Creates an image OCR extractor (if feature enabled).
    #[cfg(feature = "ocr")]
    pub fn image(language: &str) -> Box<dyn ContentExtractor> {
        Box::new(image::ImageExtractor::with_language(language))
    }
}

// Re-exports
pub use text::TextExtractor;

#[cfg(feature = "pdf")]
pub use pdf::PdfExtractor;

#[cfg(feature = "ocr")]
pub use image::ImageExtractor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_result() {
        let result = ExtractionResult::new("Hello world".to_string(), FileType::Text)
            .with_filename("test.txt".to_string());

        assert!(result.is_successful());
        assert!(!result.has_warnings());
        assert_eq!(result.text_length(), 11);
        assert_eq!(result.filename, Some("test.txt".to_string()));
    }

    #[test]
    fn test_extraction_result_warnings() {
        let mut result = ExtractionResult::new("Content".to_string(), FileType::Pdf);
        result.add_warning("Some warning".to_string());

        assert!(result.has_warnings());
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_extraction_metadata() {
        let metadata = ExtractionMetadata::new()
            .with_title("Test Doc".to_string())
            .with_author("Author".to_string())
            .with_page_count(5)
            .with_custom("key".to_string(), "value".to_string());

        assert_eq!(metadata.title, Some("Test Doc".to_string()));
        assert_eq!(metadata.author, Some("Author".to_string()));
        assert_eq!(metadata.page_count, Some(5));
        assert_eq!(metadata.custom.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_factory_text() {
        let extractor = ExtractorFactory::text();
        assert!(extractor.supports(FileType::Text));
        assert!(!extractor.supports(FileType::Pdf));
    }

    #[test]
    fn test_factory_create() {
        let text_ext = ExtractorFactory::create(FileType::Text);
        assert!(text_ext.is_some());

        let unknown_ext = ExtractorFactory::create(FileType::Unknown);
        assert!(unknown_ext.is_none());
    }
}
