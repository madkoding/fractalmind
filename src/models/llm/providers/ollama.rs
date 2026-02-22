#![allow(dead_code)]

use crate::models::llm::traits_llm::{
    ChatMessage, ChatProvider, ChatResponse, ChatRole, EmbeddingProvider, EmbeddingResponse,
    SummarizerProvider,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, warn};

/// Cliente Ollama para embeddings (local o cloud con API key)
pub struct OllamaEmbedding {
    client: Client,
    base_url: String,
    model_name: String,
    dimension: usize,
    api_key: Option<String>,
}

impl OllamaEmbedding {
    pub fn new(base_url: String, model_name: String, dimension: usize) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            dimension,
            api_key: None,
        }
    }

    pub fn with_api_key(base_url: String, model_name: String, dimension: usize, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            dimension,
            api_key: Some(api_key),
        }
    }
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbedding {
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse> {
        let start = Instant::now();
        let url = format!("{}/api/embeddings", self.base_url);

        let request = OllamaEmbedRequest {
            model: self.model_name.clone(),
            prompt: text.to_string(),
        };

        debug!("Sending embedding request to Ollama: {}", url);

        let mut request_builder = self
            .client
            .post(&url)
            .json(&request);

        if let Some(ref api_key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request_builder
            .send()
            .await
            .context("Failed to send embedding request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Ollama embedding request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let ollama_response: OllamaEmbedResponse = response
            .json()
            .await
            .context("Failed to parse Ollama embedding response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(EmbeddingResponse {
            embedding: ollama_response.embedding,
            dimension: self.dimension,
            model: self.model_name.clone(),
            latency_ms,
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResponse>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        let mut request_builder = self.client.get(&url);
        
        if let Some(ref api_key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
        }
        
        match request_builder.send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("Ollama health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

/// Cliente Ollama para chat
pub struct OllamaChat {
    client: Client,
    base_url: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    api_key: Option<String>,
}

impl OllamaChat {
    pub fn new(
        base_url: String,
        model_name: String,
        temperature: f32,
        max_tokens: u32,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            temperature,
            max_tokens,
            api_key: None,
        }
    }

    pub fn with_api_key(
        base_url: String,
        model_name: String,
        temperature: f32,
        max_tokens: u32,
        api_key: String,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            temperature,
            max_tokens,
            api_key: Some(api_key),
        }
    }
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    options: OllamaChatOptions,
}

#[derive(Debug, Serialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OllamaChatOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessageResponse,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaChatMessageResponse {
    content: String,
}

#[async_trait]
impl ChatProvider for OllamaChat {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse> {
        let start = Instant::now();
        let url = format!("{}/api/chat", self.base_url);

        let ollama_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .map(|msg| OllamaChatMessage {
                role: match msg.role {
                    ChatRole::System => "system".to_string(),
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                },
                content: msg.content.clone(),
            })
            .collect();

        let request = OllamaChatRequest {
            model: self.model_name.clone(),
            messages: ollama_messages,
            stream: false,
            options: OllamaChatOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            },
        };

        debug!("Sending chat request to Ollama: {}", url);

        let mut request_builder = self
            .client
            .post(&url)
            .json(&request);

        if let Some(ref api_key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request_builder
            .send()
            .await
            .context("Failed to send chat request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Ollama chat request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let ollama_response: OllamaChatResponse = response
            .json()
            .await
            .context("Failed to parse Ollama chat response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(ChatResponse {
            content: ollama_response.message.content,
            tokens_used: None, // Ollama no retorna tokens usados en la respuesta estándar
            latency_ms,
            model: self.model_name.clone(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        let mut request_builder = self.client.get(&url);
        
        if let Some(ref api_key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
        }
        
        match request_builder.send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("Ollama health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

/// Cliente Ollama para resumir (fase REM)
pub struct OllamaSummarizer {
    client: Client,
    base_url: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    api_key: Option<String>,
}

impl OllamaSummarizer {
    pub fn new(
        base_url: String,
        model_name: String,
        temperature: f32,
        max_tokens: u32,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            temperature,
            max_tokens,
            api_key: None,
        }
    }

    pub fn with_api_key(
        base_url: String,
        model_name: String,
        temperature: f32,
        max_tokens: u32,
        api_key: String,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            temperature,
            max_tokens,
            api_key: Some(api_key),
        }
    }

    /// Prompt por defecto para resumir
    fn default_summarize_prompt() -> &'static str {
        "Eres un sistema de resumen experto. Resume el siguiente texto de manera concisa y precisa, \
        capturando los puntos clave y la información más importante. El resumen debe ser claro, \
        coherente y mantener el significado original.\n\nTexto a resumir:\n"
    }
}

#[async_trait]
impl SummarizerProvider for OllamaSummarizer {
    async fn summarize(&self, text: &str) -> Result<String> {
        self.summarize_with_prompt(text, Self::default_summarize_prompt())
            .await
    }

    async fn summarize_with_prompt(&self, text: &str, custom_prompt: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);

        let messages = vec![
            OllamaChatMessage {
                role: "system".to_string(),
                content: "Eres un asistente especializado en crear resúmenes concisos y precisos."
                    .to_string(),
            },
            OllamaChatMessage {
                role: "user".to_string(),
                content: format!("{}\n\n{}", custom_prompt, text),
            },
        ];

        let request = OllamaChatRequest {
            model: self.model_name.clone(),
            messages,
            stream: false,
            options: OllamaChatOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            },
        };

        debug!("Sending summarization request to Ollama");

        let mut request_builder = self
            .client
            .post(&url)
            .json(&request);

        if let Some(ref api_key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request_builder
            .send()
            .await
            .context("Failed to send summarization request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Ollama summarization request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let ollama_response: OllamaChatResponse = response
            .json()
            .await
            .context("Failed to parse Ollama summarization response")?;

        Ok(ollama_response.message.content)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        let mut request_builder = self.client.get(&url);
        
        if let Some(ref api_key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
        }
        
        match request_builder.send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("Ollama health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_embedding_creation() {
        let embedding = OllamaEmbedding::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            768,
        );
        assert_eq!(embedding.dimension(), 768);
        assert_eq!(embedding.model_name(), "nomic-embed-text");
    }

    #[test]
    fn test_ollama_chat_creation() {
        let chat = OllamaChat::new(
            "http://localhost:11434".to_string(),
            "llama2".to_string(),
            0.7,
            2048,
        );
        assert_eq!(chat.model_name(), "llama2");
    }

    #[tokio::test]
    #[ignore] // Requiere Ollama corriendo
    async fn test_ollama_health_check() {
        let embedding = OllamaEmbedding::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            768,
        );
        let health = embedding.health_check().await;
        assert!(health.is_ok());
    }
}
