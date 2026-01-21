//! PDF content extractor.
//!
//! This module requires the `pdf` feature to be enabled.

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use super::{ContentExtractor, ExtractionMetadata, ExtractionResult, FileType};

/// Extractor for PDF documents.
pub struct PdfExtractor {
    /// Maximum file size to process.
    max_size: usize,
    /// Whether to preserve layout/formatting.
    preserve_layout: bool,
    /// Maximum pages to extract (0 = all).
    max_pages: usize,
}

impl PdfExtractor {
    /// Creates a new PDF extractor with default settings.
    pub fn new() -> Self {
        Self {
            max_size: 100 * 1024 * 1024, // 100MB
            preserve_layout: false,
            max_pages: 0,
        }
    }

    /// Sets the maximum file size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Sets whether to preserve layout.
    pub fn with_preserve_layout(mut self, preserve: bool) -> Self {
        self.preserve_layout = preserve;
        self
    }

    /// Sets the maximum pages to extract.
    pub fn with_max_pages(mut self, max: usize) -> Self {
        self.max_pages = max;
        self
    }

    /// Extracts text from PDF bytes using pdf-extract.
    fn extract_pdf_text(&self, data: &[u8]) -> Result<(String, ExtractionMetadata)> {
        // Use pdf-extract crate
        let text = pdf_extract::extract_text_from_mem(data)
            .map_err(|e| anyhow!("PDF extraction failed: {}", e))?;

        // Clean up the extracted text
        let cleaned = self.clean_text(&text);

        // Create metadata
        let metadata = ExtractionMetadata::new()
            .with_file_size(data.len())
            .with_mime_type("application/pdf".to_string());

        Ok((cleaned, metadata))
    }

    /// Cleans up extracted PDF text.
    fn clean_text(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_was_space = false;
        let mut prev_was_newline = false;

        for c in text.chars() {
            if c == '\n' {
                if !prev_was_newline {
                    result.push(c);
                    prev_was_newline = true;
                    prev_was_space = false;
                }
            } else if c.is_whitespace() {
                if !prev_was_space && !prev_was_newline {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else if c.is_control() {
                // Skip control characters
                continue;
            } else {
                result.push(c);
                prev_was_space = false;
                prev_was_newline = false;
            }
        }

        result.trim().to_string()
    }
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentExtractor for PdfExtractor {
    async fn extract(&self, data: &[u8]) -> Result<ExtractionResult> {
        if data.is_empty() {
            return Err(anyhow!("Empty input data"));
        }

        if data.len() > self.max_size {
            return Err(anyhow!(
                "File size {} exceeds maximum {}",
                data.len(),
                self.max_size
            ));
        }

        // Verify PDF magic bytes
        if !data.starts_with(b"%PDF") {
            return Err(anyhow!("Invalid PDF file (missing magic bytes)"));
        }

        let (text, metadata) = self.extract_pdf_text(data)?;

        let mut result = ExtractionResult::new(text, FileType::Pdf).with_metadata(metadata);

        if result.text.is_empty() {
            result.add_warning("PDF appears to contain no extractable text (may be scanned/image-based)".to_string());
        }

        Ok(result)
    }

    fn supported_types(&self) -> Vec<FileType> {
        vec![FileType::Pdf]
    }

    fn name(&self) -> &str {
        "PdfExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_extractor_creation() {
        let extractor = PdfExtractor::new();
        assert_eq!(extractor.max_size, 100 * 1024 * 1024);
        assert!(!extractor.preserve_layout);
    }

    #[test]
    fn test_pdf_extractor_builder() {
        let extractor = PdfExtractor::new()
            .with_max_size(50 * 1024 * 1024)
            .with_preserve_layout(true)
            .with_max_pages(10);

        assert_eq!(extractor.max_size, 50 * 1024 * 1024);
        assert!(extractor.preserve_layout);
        assert_eq!(extractor.max_pages, 10);
    }

    #[test]
    fn test_supported_types() {
        let extractor = PdfExtractor::new();
        assert!(extractor.supports(FileType::Pdf));
        assert!(!extractor.supports(FileType::Text));
        assert!(!extractor.supports(FileType::Image));
    }

    #[test]
    fn test_extractor_name() {
        let extractor = PdfExtractor::new();
        assert_eq!(extractor.name(), "PdfExtractor");
    }

    #[tokio::test]
    async fn test_extract_empty_fails() {
        let extractor = PdfExtractor::new();
        let result = extractor.extract(&[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_invalid_pdf_fails() {
        let extractor = PdfExtractor::new();
        let result = extractor.extract(b"Not a PDF file").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_clean_text() {
        let extractor = PdfExtractor::new();
        let dirty = "Hello   World\n\n\n\nNext\x00line";
        let cleaned = extractor.clean_text(dirty);

        assert!(!cleaned.contains("   "));
        assert!(!cleaned.contains("\n\n\n"));
        assert!(!cleaned.contains('\x00'));
    }
}
