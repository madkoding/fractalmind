//! Text chunking utilities for document processing.
//!
//! Splits large documents into smaller chunks suitable for embedding generation
//! while maintaining semantic coherence.

#![allow(dead_code)]

use super::config::IngestionConfig;

/// A chunk of text from a document.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk text content.
    pub content: String,

    /// Index of this chunk (0-based).
    pub index: usize,

    /// Total number of chunks.
    pub total: usize,

    /// Character offset in original document.
    pub start_offset: usize,

    /// Character end offset in original document.
    pub end_offset: usize,

    /// Source document identifier.
    pub source: Option<String>,
}

impl TextChunk {
    /// Creates a new text chunk.
    pub fn new(content: String, index: usize, total: usize, start: usize, end: usize) -> Self {
        Self {
            content,
            index,
            total,
            start_offset: start,
            end_offset: end,
            source: None,
        }
    }

    /// Sets the source document identifier.
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Gets the length of the chunk content.
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Checks if the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Checks if this is the first chunk.
    pub fn is_first(&self) -> bool {
        self.index == 0
    }

    /// Checks if this is the last chunk.
    pub fn is_last(&self) -> bool {
        self.index == self.total - 1
    }
}

/// Result of a chunking operation.
#[derive(Debug)]
pub struct ChunkingResult {
    /// The generated chunks.
    pub chunks: Vec<TextChunk>,

    /// Original document length in characters.
    pub original_length: usize,

    /// Average chunk size.
    pub avg_chunk_size: usize,
}

impl ChunkingResult {
    /// Gets the number of chunks.
    pub fn count(&self) -> usize {
        self.chunks.len()
    }

    /// Checks if the document was split into multiple chunks.
    pub fn was_split(&self) -> bool {
        self.chunks.len() > 1
    }
}

/// Text chunker with configurable parameters.
pub struct TextChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk_size: usize,
}

impl TextChunker {
    /// Creates a new chunker with the given parameters.
    pub fn new(chunk_size: usize, chunk_overlap: usize, min_chunk_size: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(50),
            chunk_overlap: chunk_overlap.min(chunk_size / 2),
            min_chunk_size: min_chunk_size.min(chunk_size),
        }
    }

    /// Creates a chunker from configuration.
    pub fn from_config(config: &IngestionConfig) -> Self {
        Self::new(config.chunk_size, config.chunk_overlap, config.min_chunk_size)
    }

    /// Creates a chunker with default parameters.
    pub fn with_defaults() -> Self {
        Self::new(1000, 200, 100)
    }

    /// Splits text into chunks.
    pub fn chunk(&self, text: &str) -> ChunkingResult {
        let text = text.trim();
        let original_length = text.len();

        if text.is_empty() {
            return ChunkingResult {
                chunks: vec![],
                original_length: 0,
                avg_chunk_size: 0,
            };
        }

        // If text is smaller than chunk size, return single chunk
        if text.len() <= self.chunk_size {
            return ChunkingResult {
                chunks: vec![TextChunk::new(text.to_string(), 0, 1, 0, text.len())],
                original_length,
                avg_chunk_size: text.len(),
            };
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            // Ensure start is at a valid UTF-8 boundary
            start = self.find_char_boundary(text, start);
            
            if start >= text.len() {
                break;
            }

            // Calculate end position
            let end = (start + self.chunk_size).min(text.len());

            // Try to find a good break point (sentence or word boundary)
            let actual_end = self.find_break_point(text, start, end);

            // Ensure actual_end is at a valid UTF-8 boundary
            let actual_end = self.find_char_boundary(text, actual_end);

            // Safety check: ensure we have a valid range
            if actual_end <= start {
                start = self.find_next_char_boundary(text, start + 1);
                continue;
            }

            // Extract chunk
            let chunk_text = &text[start..actual_end];

            // Skip if chunk is too small and not the last one
            if chunk_text.len() >= self.min_chunk_size || actual_end >= text.len() {
                chunks.push(TextChunk::new(
                    chunk_text.trim().to_string(),
                    chunks.len(),
                    0, // Will be updated later
                    start,
                    actual_end,
                ));
            }

            // Move start position, accounting for overlap
            if actual_end >= text.len() {
                break;
            }

            start = actual_end.saturating_sub(self.chunk_overlap);
            // Ensure start is at a valid UTF-8 boundary after overlap adjustment
            start = self.find_char_boundary(text, start);

            // Ensure we make progress
            if start >= actual_end {
                start = actual_end;
            }
        }

        // Update total count in all chunks
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total = total;
        }

        let avg_size = if chunks.is_empty() {
            0
        } else {
            chunks.iter().map(|c| c.len()).sum::<usize>() / chunks.len()
        };

        ChunkingResult {
            chunks,
            original_length,
            avg_chunk_size: avg_size,
        }
    }

    /// Finds a good break point near the target position.
    fn find_break_point(&self, text: &str, start: usize, target_end: usize) -> usize {
        if target_end >= text.len() {
            return text.len();
        }

        // Search window for break point
        let search_start = target_end.saturating_sub(100);
        let search_end = (target_end + 50).min(text.len());

        // Ensure we slice at valid UTF-8 boundaries
        let search_start = self.find_char_boundary(text, search_start);
        let search_end = self.find_char_boundary(text, search_end);

        let search_text = &text[search_start..search_end];

        // Priority 1: Find sentence boundary (. ! ? followed by space or newline)
        let sentence_breaks = [". ", ".\n", "! ", "!\n", "? ", "?\n"];
        for &break_pattern in &sentence_breaks {
            if let Some(pos) = search_text.rfind(break_pattern) {
                let abs_pos = search_start + pos + break_pattern.len();
                if abs_pos > start + self.min_chunk_size {
                    return abs_pos;
                }
            }
        }

        // Priority 2: Find paragraph boundary
        if let Some(pos) = search_text.rfind("\n\n") {
            let abs_pos = search_start + pos + 2;
            if abs_pos > start + self.min_chunk_size {
                return abs_pos;
            }
        }

        // Priority 3: Find line boundary
        if let Some(pos) = search_text.rfind('\n') {
            let abs_pos = search_start + pos + 1;
            if abs_pos > start + self.min_chunk_size {
                return abs_pos;
            }
        }

        // Priority 4: Find word boundary (space)
        if let Some(pos) = search_text.rfind(' ') {
            let abs_pos = search_start + pos + 1;
            if abs_pos > start + self.min_chunk_size {
                return abs_pos;
            }
        }

        // Fallback: use target end (ensure UTF-8 boundary)
        self.find_char_boundary(text, target_end)
    }

    /// Chunks text by sentences first, then combines into appropriately sized chunks.
    pub fn chunk_by_sentences(&self, text: &str) -> ChunkingResult {
        let text = text.trim();
        let original_length = text.len();

        if text.is_empty() {
            return ChunkingResult {
                chunks: vec![],
                original_length: 0,
                avg_chunk_size: 0,
            };
        }

        // Split into sentences
        let sentences = self.split_sentences(text);

        if sentences.is_empty() {
            return self.chunk(text);
        }

        // Combine sentences into chunks
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = 0;

        for sentence in &sentences {
            let sentence_trimmed = sentence.trim();
            if sentence_trimmed.is_empty() {
                continue;
            }

            // Check if adding this sentence would exceed chunk size
            if !current_chunk.is_empty()
                && current_chunk.len() + sentence_trimmed.len() + 1 > self.chunk_size
            {
                // Save current chunk
                let end_offset = current_start + current_chunk.len();
                chunks.push(TextChunk::new(
                    current_chunk.clone(),
                    chunks.len(),
                    0,
                    current_start,
                    end_offset,
                ));
                current_start = end_offset.saturating_sub(self.chunk_overlap);
                current_chunk = String::new();
            }

            // Add sentence to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence_trimmed);
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() && current_chunk.len() >= self.min_chunk_size {
            chunks.push(TextChunk::new(
                current_chunk.clone(),
                chunks.len(),
                0,
                current_start,
                current_start + current_chunk.len(),
            ));
        }

        // Update totals
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total = total;
        }

        let avg_size = if chunks.is_empty() {
            0
        } else {
            chunks.iter().map(|c| c.len()).sum::<usize>() / chunks.len()
        };

        ChunkingResult {
            chunks,
            original_length,
            avg_chunk_size: avg_size,
        }
    }

    /// Splits text into sentences.
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            current.push(c);

            // Check for sentence end
            if (c == '.' || c == '!' || c == '?')
                && (i + 1 >= chars.len() || chars[i + 1].is_whitespace())
            {
                // Handle abbreviations and decimals
                let is_abbreviation = self.is_likely_abbreviation(&current);
                let is_decimal = c == '.'
                    && i + 1 < chars.len()
                    && chars[i + 1].is_ascii_digit();

                if !is_abbreviation && !is_decimal {
                    sentences.push(current.trim().to_string());
                    current = String::new();
                }
            }

            i += 1;
        }

        // Add remaining text
        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        sentences
    }

    /// Checks if the text likely ends with an abbreviation.
    fn is_likely_abbreviation(&self, text: &str) -> bool {
        let common_abbrevs = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "i.e.", "e.g.",
            "Fig.", "fig.", "No.", "no.", "Vol.", "vol.", "p.", "pp.", "Inc.", "Ltd.", "Corp.",
        ];

        for abbrev in &common_abbrevs {
            if text.ends_with(abbrev) {
                return true;
            }
        }

        // Single letter followed by period (likely initial)
        let trimmed = text.trim();
        if trimmed.len() >= 2 {
            let chars: Vec<char> = trimmed.chars().collect();
            let len = chars.len();
            if chars[len - 1] == '.' && chars[len - 2].is_uppercase() {
                // Check if preceded by space or start
                if len == 2 || chars[len - 3].is_whitespace() {
                    return true;
                }
            }
        }

        false
    }

    /// Finds the nearest valid UTF-8 character boundary at or before the given position.
    fn find_char_boundary(&self, text: &str, pos: usize) -> usize {
        let pos = pos.min(text.len());
        
        // Walk backwards to find a valid UTF-8 boundary
        for i in (0..=pos).rev() {
            if text.is_char_boundary(i) {
                return i;
            }
        }
        
        0
    }

    /// Finds the nearest valid UTF-8 character boundary at or after the given position.
    fn find_next_char_boundary(&self, text: &str, pos: usize) -> usize {
        let pos = pos.min(text.len());
        
        // Walk forward to find a valid UTF-8 boundary
        for i in pos..=text.len() {
            if text.is_char_boundary(i) {
                return i;
            }
        }
        
        text.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_small_text() {
        let chunker = TextChunker::with_defaults();
        let result = chunker.chunk("This is a small text.");

        assert_eq!(result.count(), 1);
        assert!(!result.was_split());
        assert_eq!(result.chunks[0].content, "This is a small text.");
    }

    #[test]
    fn test_chunk_empty_text() {
        let chunker = TextChunker::with_defaults();
        let result = chunker.chunk("");

        assert_eq!(result.count(), 0);
        assert!(!result.was_split());
    }

    #[test]
    fn test_chunk_large_text() {
        let chunker = TextChunker::new(100, 20, 50);
        let text = "Lorem ipsum dolor sit amet. ".repeat(20);
        let result = chunker.chunk(&text);

        assert!(result.was_split());
        assert!(result.count() > 1);

        // Verify chunks are correctly indexed
        for (i, chunk) in result.chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
            assert_eq!(chunk.total, result.count());
        }
    }

    #[test]
    fn test_chunk_preserves_sentence_boundaries() {
        let chunker = TextChunker::new(100, 20, 20);
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let result = chunker.chunk(text);

        // Verify no chunk starts or ends mid-sentence
        for chunk in &result.chunks {
            let content = &chunk.content;
            // Each chunk should end with a complete sentence or be the last chunk
            if !chunk.is_last() {
                assert!(
                    content.ends_with('.') || content.ends_with('!') || content.ends_with('?'),
                    "Chunk should end with sentence boundary: {}",
                    content
                );
            }
        }
    }

    #[test]
    fn test_chunk_with_overlap() {
        let chunker = TextChunker::new(50, 10, 20);
        let text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12";
        let result = chunker.chunk(text);

        if result.count() >= 2 {
            // Check that consecutive chunks have some overlap
            let chunk1 = &result.chunks[0].content;
            let chunk2 = &result.chunks[1].content;

            // Find overlapping content
            let has_overlap = chunk1.split_whitespace().any(|word| chunk2.contains(word));
            assert!(has_overlap, "Chunks should have some overlap");
        }
    }

    #[test]
    fn test_text_chunk_properties() {
        let chunk = TextChunk::new("Test content".to_string(), 2, 5, 100, 112);

        assert_eq!(chunk.len(), 12);
        assert!(!chunk.is_empty());
        assert!(!chunk.is_first());
        assert!(!chunk.is_last());

        let first = TextChunk::new("First".to_string(), 0, 3, 0, 5);
        assert!(first.is_first());

        let last = TextChunk::new("Last".to_string(), 2, 3, 90, 94);
        assert!(last.is_last());
    }

    #[test]
    fn test_chunk_with_source() {
        let chunk = TextChunk::new("Content".to_string(), 0, 1, 0, 7).with_source("doc.pdf".to_string());

        assert_eq!(chunk.source, Some("doc.pdf".to_string()));
    }

    #[test]
    fn test_chunk_by_sentences() {
        let chunker = TextChunker::new(100, 20, 20);
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let result = chunker.chunk_by_sentences(text);

        assert!(result.count() >= 1);
        for chunk in &result.chunks {
            assert!(chunk.len() <= 100 + 20); // Allow some flexibility
        }
    }

    #[test]
    fn test_abbreviation_detection() {
        let chunker = TextChunker::with_defaults();

        assert!(chunker.is_likely_abbreviation("Dr."));
        assert!(chunker.is_likely_abbreviation("etc."));
        assert!(chunker.is_likely_abbreviation("Hello Mr."));
        assert!(!chunker.is_likely_abbreviation("Hello."));
    }

    #[test]
    fn test_from_config() {
        let config = IngestionConfig::new()
            .with_chunk_size(500)
            .with_chunk_overlap(50)
            .with_min_chunk_size(100);

        let chunker = TextChunker::from_config(&config);
        let text = "Test ".repeat(200);
        let result = chunker.chunk(&text);

        assert!(result.avg_chunk_size <= 550); // chunk_size + some buffer
    }

    #[test]
    fn test_chunk_utf8_multibyte_characters() {
        // Test with Spanish text containing multi-byte UTF-8 characters
        let chunker = TextChunker::new(50, 10, 20);
        let text = "Bienvenido á la programación. Más información aquí. Años de experiencia. Niño pequeño.";
        let result = chunker.chunk(text);

        // Should not panic and produce valid chunks
        assert!(result.count() >= 1);
        for chunk in &result.chunks {
            // Each chunk should be valid UTF-8 (implicit by being a String)
            assert!(!chunk.content.is_empty() || chunk.is_last());
        }
    }

    #[test]
    fn test_chunk_utf8_mixed_content() {
        // Test with mixed content: ASCII, Spanish, Japanese, emojis
        let chunker = TextChunker::new(100, 20, 30);
        let text = "Hello world. Más información aquí. 日本語テスト。絵文字も含む 🎉🚀💻. Back to English.";
        let result = chunker.chunk(text);

        // Should not panic
        assert!(result.count() >= 1);
        
        // Verify we can iterate and access content without panics
        for chunk in &result.chunks {
            let _ = chunk.content.len();
            let _ = chunk.content.chars().count();
        }
    }
}
