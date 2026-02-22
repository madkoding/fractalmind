#![allow(dead_code)]

use anyhow::Result;
use async_trait::async_trait;

use super::traits_llm::{ChatMessage, ChatResponse, EmbeddingResponse};

/// Estrategia para usar modelos (Fractal vs Ollama)
#[async_trait]
pub trait ModelStrategy: Send + Sync {
    /// Genera embeddings usando la estrategia (batch)
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResponse>>;

    /// Genera respuesta de chat usando la estrategia
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse>;

    /// Resume texto usando la estrategia
    async fn summarize(&self, text: &str) -> Result<String>;

    /// Nombre de la estrategia
    fn name(&self) -> &str;
}

/// Estrategia que usa modelos fractales almacenados
pub struct FractalModelStrategy {
    model_id: String,
    // TODO: Añadir referencia a DB y grafo fractal
}

impl FractalModelStrategy {
    pub fn new(model_id: String) -> Self {
        Self { model_id }
    }
}

#[async_trait]
impl ModelStrategy for FractalModelStrategy {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResponse>> {
        // TODO: Implementar navegación por grafo fractal para generar embeddings
        // Por ahora, retorna embeddings vacíos
        let responses = texts
            .iter()
            .map(|_| EmbeddingResponse {
                embedding: vec![0.0; 768],
                dimension: 768,
                model: self.model_id.clone(),
                latency_ms: 0,
            })
            .collect();
        Ok(responses)
    }

    async fn chat(&self, _messages: Vec<ChatMessage>) -> Result<ChatResponse> {
        // TODO: Implementar generación usando grafo fractal
        Ok(ChatResponse {
            content: "Fractal model response (not implemented yet)".to_string(),
            model: self.model_id.clone(),
            tokens_used: None,
            latency_ms: 0,
        })
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        // TODO: Implementar resumen usando grafo fractal
        Ok(format!("Summary of: {}...", &text[..text.len().min(50)]))
    }

    fn name(&self) -> &str {
        "FractalModel"
    }
}

/// Estrategia que usa Ollama directamente
pub struct OllamaModelStrategy {
    base_url: String,
    model_name: String,
    api_key: Option<String>,
}

impl OllamaModelStrategy {
    pub fn new(base_url: String, model_name: String) -> Self {
        Self { base_url, model_name, api_key: None }
    }

    pub fn with_api_key(base_url: String, model_name: String, api_key: String) -> Self {
        Self { base_url, model_name, api_key: Some(api_key) }
    }
}

#[async_trait]
impl ModelStrategy for OllamaModelStrategy {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResponse>> {
        // Usar el provider de Ollama existente
        use super::providers::OllamaEmbedding;
        use super::traits_llm::EmbeddingProvider;
        
        let provider = if let Some(key) = &self.api_key {
            OllamaEmbedding::with_api_key(
                self.base_url.clone(),
                self.model_name.clone(),
                768,
                key.clone(),
            )
        } else {
            OllamaEmbedding::new(self.base_url.clone(), self.model_name.clone(), 768)
        };
        provider.embed_batch(&texts).await
    }

    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse> {
        use super::providers::OllamaChat;
        use super::traits_llm::ChatProvider;
        
        let provider = if let Some(key) = &self.api_key {
            OllamaChat::with_api_key(
                self.base_url.clone(),
                self.model_name.clone(),
                0.7, // temperature
                2048, // max_tokens
                key.clone(),
            )
        } else {
            OllamaChat::new(
                self.base_url.clone(),
                self.model_name.clone(),
                0.7, // temperature
                2048, // max_tokens
            )
        };
        provider.chat(&messages).await
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        use super::providers::OllamaSummarizer;
        use super::traits_llm::SummarizerProvider;
        
        let provider = if let Some(key) = &self.api_key {
            OllamaSummarizer::with_api_key(
                self.base_url.clone(),
                self.model_name.clone(),
                0.3, // temperature
                512, // max_tokens
                key.clone(),
            )
        } else {
            OllamaSummarizer::new(
                self.base_url.clone(),
                self.model_name.clone(),
                0.3, // temperature
                512, // max_tokens
            )
        };
        provider.summarize(text).await
    }

    fn name(&self) -> &str {
        "Ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_strategy_creation() {
        let strategy = FractalModelStrategy::new("model:123".to_string());
        assert_eq!(strategy.name(), "FractalModel");
    }

    #[test]
    fn test_ollama_strategy_creation() {
        let strategy = OllamaModelStrategy::new(
            "http://localhost:11434".to_string(),
            "llama2".to_string(),
        );
        assert_eq!(strategy.name(), "Ollama");
    }
}
