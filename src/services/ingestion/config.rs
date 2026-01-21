//! Ingestion service configuration.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Supported file types for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileType {
    /// Plain text files (.txt, .md, .rst)
    Text,
    /// PDF documents (.pdf)
    Pdf,
    /// Image files (.png, .jpg, .jpeg, .webp, .tiff)
    Image,
    /// Unknown/unsupported file type
    Unknown,
}

impl FileType {
    /// Detects file type from extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "txt" | "md" | "rst" | "text" | "markdown" => Self::Text,
            "pdf" => Self::Pdf,
            "png" | "jpg" | "jpeg" | "webp" | "tiff" | "bmp" | "gif" => Self::Image,
            _ => Self::Unknown,
        }
    }

    /// Detects file type from MIME type.
    pub fn from_mime(mime: &str) -> Self {
        match mime {
            "text/plain" | "text/markdown" | "text/x-rst" => Self::Text,
            "application/pdf" => Self::Pdf,
            m if m.starts_with("image/") => Self::Image,
            _ => Self::Unknown,
        }
    }

    /// Gets the display name for this file type.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Pdf => "PDF",
            Self::Image => "Image",
            Self::Unknown => "Unknown",
        }
    }

    /// Checks if OCR is needed for this file type.
    pub fn needs_ocr(&self) -> bool {
        matches!(self, Self::Image)
    }

    /// Checks if this file type is supported.
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Configuration for the ingestion service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Maximum file size in bytes (default: 50MB).
    pub max_file_size: usize,

    /// Chunk size for splitting large documents (in characters).
    pub chunk_size: usize,

    /// Overlap between chunks (in characters).
    pub chunk_overlap: usize,

    /// Minimum chunk size (smaller chunks are merged).
    pub min_chunk_size: usize,

    /// Whether to enable OCR for images.
    pub enable_ocr: bool,

    /// OCR language (e.g., "eng", "spa", "deu").
    pub ocr_language: String,

    /// Whether to enable PDF extraction.
    pub enable_pdf: bool,

    /// Whether to preserve formatting in extracted text.
    pub preserve_formatting: bool,

    /// Supported file extensions.
    pub allowed_extensions: Vec<String>,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            max_file_size: 50 * 1024 * 1024, // 50MB
            chunk_size: 1000,
            chunk_overlap: 200,
            min_chunk_size: 100,
            enable_ocr: true,
            ocr_language: "eng".to_string(),
            enable_pdf: true,
            preserve_formatting: false,
            allowed_extensions: vec![
                "txt".to_string(),
                "md".to_string(),
                "pdf".to_string(),
                "png".to_string(),
                "jpg".to_string(),
                "jpeg".to_string(),
            ],
        }
    }
}

impl IngestionConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set maximum file size.
    pub fn with_max_file_size(mut self, size: usize) -> Self {
        self.max_file_size = size;
        self
    }

    /// Builder: set chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(100);
        self
    }

    /// Builder: set chunk overlap.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap.min(self.chunk_size / 2);
        self
    }

    /// Builder: set minimum chunk size.
    pub fn with_min_chunk_size(mut self, size: usize) -> Self {
        self.min_chunk_size = size;
        self
    }

    /// Builder: enable/disable OCR.
    pub fn with_ocr(mut self, enabled: bool) -> Self {
        self.enable_ocr = enabled;
        self
    }

    /// Builder: set OCR language.
    pub fn with_ocr_language(mut self, lang: &str) -> Self {
        self.ocr_language = lang.to_string();
        self
    }

    /// Builder: enable/disable PDF extraction.
    pub fn with_pdf(mut self, enabled: bool) -> Self {
        self.enable_pdf = enabled;
        self
    }

    /// Builder: set formatting preservation.
    pub fn with_preserve_formatting(mut self, preserve: bool) -> Self {
        self.preserve_formatting = preserve;
        self
    }

    /// Builder: set allowed extensions.
    pub fn with_allowed_extensions(mut self, extensions: Vec<String>) -> Self {
        self.allowed_extensions = extensions;
        self
    }

    /// Creates configuration from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("INGESTION_MAX_FILE_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                config.max_file_size = size;
            }
        }

        if let Ok(val) = std::env::var("INGESTION_CHUNK_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                config.chunk_size = size;
            }
        }

        if let Ok(val) = std::env::var("INGESTION_CHUNK_OVERLAP") {
            if let Ok(overlap) = val.parse::<usize>() {
                config.chunk_overlap = overlap;
            }
        }

        if let Ok(val) = std::env::var("INGESTION_OCR_ENABLED") {
            config.enable_ocr = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = std::env::var("INGESTION_OCR_LANGUAGE") {
            config.ocr_language = val;
        }

        if let Ok(val) = std::env::var("INGESTION_PDF_ENABLED") {
            config.enable_pdf = val.to_lowercase() == "true" || val == "1";
        }

        config
    }

    /// Checks if a file extension is allowed.
    pub fn is_extension_allowed(&self, ext: &str) -> bool {
        self.allowed_extensions.iter().any(|e| e.eq_ignore_ascii_case(ext))
    }

    /// Checks if a file size is within limits.
    pub fn is_size_allowed(&self, size: usize) -> bool {
        size <= self.max_file_size
    }

    /// Validates configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.chunk_size < 50 {
            return Err(ConfigError::ChunkSizeTooSmall);
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(ConfigError::OverlapTooLarge);
        }
        if self.min_chunk_size > self.chunk_size {
            return Err(ConfigError::MinChunkTooLarge);
        }
        Ok(())
    }
}

/// Configuration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    ChunkSizeTooSmall,
    OverlapTooLarge,
    MinChunkTooLarge,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChunkSizeTooSmall => write!(f, "Chunk size must be at least 50 characters"),
            Self::OverlapTooLarge => write!(f, "Chunk overlap must be less than chunk size"),
            Self::MinChunkTooLarge => write!(f, "Minimum chunk size must be less than chunk size"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_type_from_extension() {
        assert_eq!(FileType::from_extension("txt"), FileType::Text);
        assert_eq!(FileType::from_extension("MD"), FileType::Text);
        assert_eq!(FileType::from_extension("pdf"), FileType::Pdf);
        assert_eq!(FileType::from_extension("PNG"), FileType::Image);
        assert_eq!(FileType::from_extension("jpg"), FileType::Image);
        assert_eq!(FileType::from_extension("xyz"), FileType::Unknown);
    }

    #[test]
    fn test_file_type_from_mime() {
        assert_eq!(FileType::from_mime("text/plain"), FileType::Text);
        assert_eq!(FileType::from_mime("application/pdf"), FileType::Pdf);
        assert_eq!(FileType::from_mime("image/png"), FileType::Image);
        assert_eq!(FileType::from_mime("image/jpeg"), FileType::Image);
        assert_eq!(FileType::from_mime("application/octet-stream"), FileType::Unknown);
    }

    #[test]
    fn test_file_type_needs_ocr() {
        assert!(!FileType::Text.needs_ocr());
        assert!(!FileType::Pdf.needs_ocr());
        assert!(FileType::Image.needs_ocr());
    }

    #[test]
    fn test_config_default() {
        let config = IngestionConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.chunk_overlap, 200);
        assert!(config.enable_ocr);
        assert!(config.enable_pdf);
    }

    #[test]
    fn test_config_builder() {
        let config = IngestionConfig::new()
            .with_chunk_size(500)
            .with_chunk_overlap(100)
            .with_ocr(false)
            .with_ocr_language("spa");

        assert_eq!(config.chunk_size, 500);
        assert_eq!(config.chunk_overlap, 100);
        assert!(!config.enable_ocr);
        assert_eq!(config.ocr_language, "spa");
    }

    #[test]
    fn test_config_validation() {
        let config = IngestionConfig::new();
        assert!(config.validate().is_ok());

        // Test validation with chunk size below minimum (bypass builder's enforcement)
        let mut config = IngestionConfig::new();
        config.chunk_size = 30;
        assert!(matches!(config.validate(), Err(ConfigError::ChunkSizeTooSmall)));

        // Test overlap too large
        let mut config = IngestionConfig::new();
        config.chunk_overlap = config.chunk_size + 1;
        assert!(matches!(config.validate(), Err(ConfigError::OverlapTooLarge)));

        // Test min chunk too large
        let mut config = IngestionConfig::new();
        config.min_chunk_size = config.chunk_size + 1;
        assert!(matches!(config.validate(), Err(ConfigError::MinChunkTooLarge)));
    }

    #[test]
    fn test_extension_allowed() {
        let config = IngestionConfig::default();
        assert!(config.is_extension_allowed("txt"));
        assert!(config.is_extension_allowed("TXT"));
        assert!(config.is_extension_allowed("pdf"));
        assert!(!config.is_extension_allowed("exe"));
    }

    #[test]
    fn test_size_allowed() {
        let config = IngestionConfig::new().with_max_file_size(1024);
        assert!(config.is_size_allowed(500));
        assert!(config.is_size_allowed(1024));
        assert!(!config.is_size_allowed(1025));
    }
}
