//! Plain text content extractor.

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use super::{ContentExtractor, ExtractionMetadata, ExtractionResult, FileType};

/// Extractor for plain text files (.txt, .md, .rst).
pub struct TextExtractor {
    /// Maximum file size to process.
    max_size: usize,
    /// Normalize line endings.
    normalize_line_endings: bool,
    /// Strip extra whitespace.
    strip_whitespace: bool,
}

impl TextExtractor {
    /// Creates a new text extractor with default settings.
    pub fn new() -> Self {
        Self {
            max_size: 50 * 1024 * 1024, // 50MB
            normalize_line_endings: true,
            strip_whitespace: true,
        }
    }

    /// Sets the maximum file size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Sets whether to normalize line endings.
    pub fn with_normalize_line_endings(mut self, normalize: bool) -> Self {
        self.normalize_line_endings = normalize;
        self
    }

    /// Sets whether to strip extra whitespace.
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Detects text encoding from BOM or content.
    fn detect_encoding(&self, data: &[u8]) -> &'static str {
        // Check for BOM
        if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
            return "utf-8";
        }
        if data.starts_with(&[0xFF, 0xFE]) {
            return "utf-16-le";
        }
        if data.starts_with(&[0xFE, 0xFF]) {
            return "utf-16-be";
        }

        // Default to UTF-8
        "utf-8"
    }

    /// Removes BOM from the beginning of text.
    fn remove_bom<'a>(&self, text: &'a str) -> &'a str {
        text.strip_prefix('\u{FEFF}').unwrap_or(text)
    }

    /// Normalizes line endings to Unix style (\n).
    fn normalize_endings(&self, text: &str) -> String {
        text.replace("\r\n", "\n").replace('\r', "\n")
    }

    /// Strips excessive whitespace.
    fn strip_excessive_whitespace(&self, text: &str) -> String {
        // Replace multiple spaces with single space
        let mut result = String::with_capacity(text.len());
        let mut prev_was_space = false;
        let mut prev_was_newline = false;

        for c in text.chars() {
            if c == '\n' {
                // Allow up to 2 consecutive newlines (one blank line)
                if !prev_was_newline {
                    result.push(c);
                    prev_was_newline = true;
                    prev_was_space = false;
                } else if result.chars().last() != Some('\n') {
                    result.push(c);
                }
            } else if c.is_whitespace() {
                if !prev_was_space && !prev_was_newline {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(c);
                prev_was_space = false;
                prev_was_newline = false;
            }
        }

        result.trim().to_string()
    }

    /// Extracts text content from bytes.
    fn extract_text(&self, data: &[u8]) -> Result<String> {
        let encoding = self.detect_encoding(data);

        let text = match encoding {
            "utf-8" => {
                // Skip BOM if present
                let start = if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
                    3
                } else {
                    0
                };
                String::from_utf8_lossy(&data[start..]).to_string()
            }
            "utf-16-le" => {
                // Skip BOM and decode UTF-16 LE
                let words: Vec<u16> = data[2..]
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                String::from_utf16_lossy(&words)
            }
            "utf-16-be" => {
                // Skip BOM and decode UTF-16 BE
                let words: Vec<u16> = data[2..]
                    .chunks_exact(2)
                    .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
                    .collect();
                String::from_utf16_lossy(&words)
            }
            _ => String::from_utf8_lossy(data).to_string(),
        };

        let mut processed = text;

        if self.normalize_line_endings {
            processed = self.normalize_endings(&processed);
        }

        if self.strip_whitespace {
            processed = self.strip_excessive_whitespace(&processed);
        }

        Ok(processed)
    }
}

impl Default for TextExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentExtractor for TextExtractor {
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

        let text = self.extract_text(data)?;

        let metadata = ExtractionMetadata::new()
            .with_file_size(data.len())
            .with_mime_type("text/plain".to_string());

        let mut result = ExtractionResult::new(text, FileType::Text).with_metadata(metadata);

        // Add warning if lossy conversion occurred
        if String::from_utf8(data.to_vec()).is_err() {
            result.add_warning("Some characters were replaced during encoding conversion".to_string());
        }

        Ok(result)
    }

    fn supported_types(&self) -> Vec<FileType> {
        vec![FileType::Text]
    }

    fn name(&self) -> &str {
        "TextExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_simple_text() {
        let extractor = TextExtractor::new();
        let data = b"Hello, World!";

        let result = extractor.extract(data).await.unwrap();

        assert_eq!(result.text, "Hello, World!");
        assert!(result.is_successful());
        assert_eq!(result.file_type, FileType::Text);
    }

    #[tokio::test]
    async fn test_extract_multiline_text() {
        let extractor = TextExtractor::new();
        let data = b"Line 1\nLine 2\nLine 3";

        let result = extractor.extract(data).await.unwrap();

        assert!(result.text.contains("Line 1"));
        assert!(result.text.contains("Line 2"));
        assert!(result.text.contains("Line 3"));
    }

    #[tokio::test]
    async fn test_normalize_line_endings() {
        let extractor = TextExtractor::new();
        let data = b"Line 1\r\nLine 2\rLine 3\nLine 4";

        let result = extractor.extract(data).await.unwrap();

        assert!(!result.text.contains("\r\n"));
        assert!(!result.text.contains('\r'));
    }

    #[tokio::test]
    async fn test_strip_excessive_whitespace() {
        let extractor = TextExtractor::new();
        let data = b"Hello    World  \n\n\n\nNext line";

        let result = extractor.extract(data).await.unwrap();

        assert!(!result.text.contains("    "));
        assert!(!result.text.contains("\n\n\n"));
    }

    #[tokio::test]
    async fn test_extract_utf8_with_bom() {
        let extractor = TextExtractor::new();
        let mut data = vec![0xEF, 0xBB, 0xBF]; // UTF-8 BOM
        data.extend_from_slice(b"Hello");

        let result = extractor.extract(&data).await.unwrap();

        assert_eq!(result.text, "Hello");
    }

    #[tokio::test]
    async fn test_extract_empty_fails() {
        let extractor = TextExtractor::new();
        let result = extractor.extract(&[]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_too_large_fails() {
        let extractor = TextExtractor::new().with_max_size(10);
        let data = b"This is more than 10 bytes";

        let result = extractor.extract(data).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_detect_encoding() {
        let extractor = TextExtractor::new();

        // UTF-8 BOM
        assert_eq!(
            extractor.detect_encoding(&[0xEF, 0xBB, 0xBF, 0x48]),
            "utf-8"
        );

        // UTF-16 LE BOM
        assert_eq!(extractor.detect_encoding(&[0xFF, 0xFE, 0x00, 0x00]), "utf-16-le");

        // UTF-16 BE BOM
        assert_eq!(extractor.detect_encoding(&[0xFE, 0xFF, 0x00, 0x00]), "utf-16-be");

        // No BOM (default UTF-8)
        assert_eq!(extractor.detect_encoding(b"Hello"), "utf-8");
    }

    #[test]
    fn test_supported_types() {
        let extractor = TextExtractor::new();
        assert!(extractor.supports(FileType::Text));
        assert!(!extractor.supports(FileType::Pdf));
        assert!(!extractor.supports(FileType::Image));
    }

    #[test]
    fn test_extractor_name() {
        let extractor = TextExtractor::new();
        assert_eq!(extractor.name(), "TextExtractor");
    }

    #[tokio::test]
    async fn test_metadata_populated() {
        let extractor = TextExtractor::new();
        let data = b"Test content";

        let result = extractor.extract(data).await.unwrap();

        assert_eq!(result.metadata.file_size, Some(12));
        assert_eq!(result.metadata.mime_type, Some("text/plain".to_string()));
    }
}
