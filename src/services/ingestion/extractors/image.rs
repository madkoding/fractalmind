//! Image OCR content extractor.
//!
//! This module requires the `ocr` feature to be enabled.

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use super::{ContentExtractor, ExtractionMetadata, ExtractionResult, FileType};

/// Extractor for images using OCR.
pub struct ImageExtractor {
    /// OCR language code (e.g., "eng", "spa", "deu").
    language: String,
    /// Maximum file size to process.
    max_size: usize,
    /// PSM (Page Segmentation Mode) for Tesseract.
    page_seg_mode: i32,
}

impl ImageExtractor {
    /// Creates a new image extractor with English language.
    pub fn new() -> Self {
        Self {
            language: "eng".to_string(),
            max_size: 50 * 1024 * 1024, // 50MB
            page_seg_mode: 3, // Fully automatic page segmentation
        }
    }

    /// Creates an image extractor with a specific language.
    pub fn with_language(language: &str) -> Self {
        Self {
            language: language.to_string(),
            ..Self::new()
        }
    }

    /// Sets the OCR language.
    pub fn set_language(mut self, language: &str) -> Self {
        self.language = language.to_string();
        self
    }

    /// Sets the maximum file size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Sets the page segmentation mode.
    ///
    /// Common modes:
    /// - 0: Orientation and script detection only
    /// - 1: Automatic page segmentation with OSD
    /// - 3: Fully automatic page segmentation (default)
    /// - 6: Assume a single uniform block of text
    /// - 7: Treat the image as a single text line
    pub fn with_page_seg_mode(mut self, mode: i32) -> Self {
        self.page_seg_mode = mode;
        self
    }

    /// Detects image format from magic bytes.
    fn detect_image_format(&self, data: &[u8]) -> Option<&'static str> {
        if data.len() < 8 {
            return None;
        }

        // PNG
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return Some("image/png");
        }

        // JPEG
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return Some("image/jpeg");
        }

        // GIF
        if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            return Some("image/gif");
        }

        // WebP
        if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
            return Some("image/webp");
        }

        // BMP
        if data.starts_with(b"BM") {
            return Some("image/bmp");
        }

        // TIFF (little-endian and big-endian)
        if data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
        {
            return Some("image/tiff");
        }

        None
    }

    /// Performs OCR on image data using Tesseract.
    fn perform_ocr(&self, data: &[u8]) -> Result<String> {
        use tesseract::Tesseract;

        // Create Tesseract instance
        let mut tess = Tesseract::new(None, Some(&self.language))
            .map_err(|e| anyhow!("Failed to initialize Tesseract: {}", e))?;

        // Set page segmentation mode
        tess = tess
            .set_variable("tessedit_pageseg_mode", &self.page_seg_mode.to_string())
            .map_err(|e| anyhow!("Failed to set PSM: {}", e))?;

        // Set image from memory
        tess = tess
            .set_image_from_mem(data)
            .map_err(|e| anyhow!("Failed to set image: {}", e))?;

        // Perform OCR
        let text = tess
            .get_text()
            .map_err(|e| anyhow!("OCR failed: {}", e))?;

        Ok(self.clean_ocr_text(&text))
    }

    /// Cleans up OCR output text.
    fn clean_ocr_text(&self, text: &str) -> String {
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
            } else if c.is_control() || c == '\u{FFFD}' {
                // Skip control characters and replacement chars
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

impl Default for ImageExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentExtractor for ImageExtractor {
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

        // Detect image format
        let mime_type = self
            .detect_image_format(data)
            .ok_or_else(|| anyhow!("Unknown or unsupported image format"))?;

        // Perform OCR
        let text = self.perform_ocr(data)?;

        let metadata = ExtractionMetadata::new()
            .with_file_size(data.len())
            .with_mime_type(mime_type.to_string())
            .with_custom("ocr_language".to_string(), self.language.clone());

        let mut result = ExtractionResult::new(text, FileType::Image).with_metadata(metadata);

        if result.text.is_empty() {
            result.add_warning("No text detected in image".to_string());
        }

        Ok(result)
    }

    fn supported_types(&self) -> Vec<FileType> {
        vec![FileType::Image]
    }

    fn name(&self) -> &str {
        "ImageExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_extractor_creation() {
        let extractor = ImageExtractor::new();
        assert_eq!(extractor.language, "eng");
        assert_eq!(extractor.page_seg_mode, 3);
    }

    #[test]
    fn test_image_extractor_with_language() {
        let extractor = ImageExtractor::with_language("spa");
        assert_eq!(extractor.language, "spa");
    }

    #[test]
    fn test_image_extractor_builder() {
        let extractor = ImageExtractor::new()
            .set_language("deu")
            .with_max_size(10 * 1024 * 1024)
            .with_page_seg_mode(6);

        assert_eq!(extractor.language, "deu");
        assert_eq!(extractor.max_size, 10 * 1024 * 1024);
        assert_eq!(extractor.page_seg_mode, 6);
    }

    #[test]
    fn test_supported_types() {
        let extractor = ImageExtractor::new();
        assert!(extractor.supports(FileType::Image));
        assert!(!extractor.supports(FileType::Text));
        assert!(!extractor.supports(FileType::Pdf));
    }

    #[test]
    fn test_extractor_name() {
        let extractor = ImageExtractor::new();
        assert_eq!(extractor.name(), "ImageExtractor");
    }

    #[test]
    fn test_detect_image_format() {
        let extractor = ImageExtractor::new();

        // PNG
        let png_header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(extractor.detect_image_format(&png_header), Some("image/png"));

        // JPEG
        let jpg_header = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(extractor.detect_image_format(&jpg_header), Some("image/jpeg"));

        // GIF
        let gif_header = b"GIF89a\x00\x00";
        assert_eq!(extractor.detect_image_format(gif_header), Some("image/gif"));

        // Unknown
        let unknown = b"unknown";
        assert_eq!(extractor.detect_image_format(unknown), None);
    }

    #[test]
    fn test_clean_ocr_text() {
        let extractor = ImageExtractor::new();
        let dirty = "Hello   World\n\n\nNext  line\x00text";
        let cleaned = extractor.clean_ocr_text(dirty);

        assert!(!cleaned.contains("   "));
        assert!(!cleaned.contains("\n\n\n"));
        assert!(!cleaned.contains('\x00'));
    }

    #[tokio::test]
    async fn test_extract_empty_fails() {
        let extractor = ImageExtractor::new();
        let result = extractor.extract(&[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_unknown_format_fails() {
        let extractor = ImageExtractor::new();
        let result = extractor.extract(b"not an image").await;
        assert!(result.is_err());
    }
}
