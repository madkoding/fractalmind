#![allow(dead_code)]

use crate::models::llm::traits_llm::{
    ChatMessage, ChatProvider, ChatResponse, ChatRole, EmbeddingProvider, EmbeddingResponse,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, warn};

const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com/v1";

/// Cliente Anthropic para embeddings
pub struct AnthropicEmbedding {
    client: Client,
    api_key: String,
    model_name: String,
    dimension: usize,
}

impl AnthropicEmbedding {
    pub fn new(api_key: String, model_name: String, dimension: usize) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model_name,
            dimension,
        }
    }
}

#[derive(Debug, Serialize)]
struct AnthropicEmbedRequest {
    input: String,
    model: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicEmbedResponse {
    data: Vec<AnthropicEmbedData>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicEmbedData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
}

#[async_trait]
impl EmbeddingProvider for AnthropicEmbedding {
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse> {
        let start = Instant::now();
        let url = format!("{}/embeddings", ANTHROPIC_API_BASE);

        let request = AnthropicEmbedRequest {
            input: text.to_string(),
            model: self.model_name.clone(),
        };

        debug!("Sending embedding request to Anthropic");

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("Anthropic-Version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send embedding request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Anthropic embedding request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let anthropic_response: AnthropicEmbedResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic embedding response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let embedding = anthropic_response
            .data
            .first()
            .context("No embedding data in Anthropic response")?
            .embedding
            .clone();

        Ok(EmbeddingResponse {
            embedding,
            dimension: self.dimension,
            model: self.model_name.clone(),
            latency_ms,
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResponse>> {
        let start = Instant::now();
        let url = format!("{}/embeddings", ANTHROPIC_API_BASE);

        #[derive(Serialize)]
        struct BatchRequest {
            input: Vec<String>,
            model: String,
        }

        let request = BatchRequest {
            input: texts.to_vec(),
            model: self.model_name.clone(),
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("Anthropic-Version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send batch embedding request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Anthropic batch embedding request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let anthropic_response: AnthropicEmbedResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic batch embedding response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let results = anthropic_response
            .data
            .into_iter()
            .map(|data| EmbeddingResponse {
                embedding: data.embedding,
                dimension: self.dimension,
                model: self.model_name.clone(),
                latency_ms,
            })
            .collect();

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", ANTHROPIC_API_BASE);
        match self
            .client
            .get(&url)
            .header("x-api-key", &self.api_key)
            .header("Anthropic-Version", "2023-06-01")
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("Anthropic health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

/// Cliente Anthropic para chat
pub struct AnthropicChat {
    client: Client,
    api_key: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
}

impl AnthropicChat {
    pub fn new(api_key: String, model_name: String, temperature: f32, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model_name,
            temperature,
            max_tokens,
        }
    }
}

#[derive(Debug, Serialize)]
struct AnthropicChatRequest {
    model: String,
    messages: Vec<AnthropicChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct AnthropicChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicChatResponse {
    content: Vec<AnthropicChatContent>,
    model: String,
    role: String,
    usage: AnthropicChatUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicChatContent {
    text: String,
    #[serde(rename = "type")]
    content_type: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicChatUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[async_trait]
impl ChatProvider for AnthropicChat {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse> {
        let start = Instant::now();
        let url = format!("{}/messages", ANTHROPIC_API_BASE);

        let anthropic_messages: Vec<AnthropicChatMessage> = messages
            .iter()
            .map(|msg| AnthropicChatMessage {
                role: match msg.role {
                    ChatRole::System => {
                        warn!("Anthropic uses 'system' role differently - placing in system prompt");
                        "user".to_string()
                    }
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                },
                content: msg.content.clone(),
            })
            .collect();

        let request = AnthropicChatRequest {
            model: self.model_name.clone(),
            messages: anthropic_messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        };

        debug!("Sending chat request to Anthropic");

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("Anthropic-Version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send chat request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Anthropic chat request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let anthropic_response: AnthropicChatResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic chat response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let content = anthropic_response
            .content
            .first()
            .context("No content in Anthropic response")?
            .text
            .clone();

        Ok(ChatResponse {
            content,
            tokens_used: Some(anthropic_response.usage.output_tokens),
            latency_ms,
            model: self.model_name.clone(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", ANTHROPIC_API_BASE);
        match self
            .client
            .get(&url)
            .header("x-api-key", &self.api_key)
            .header("Anthropic-Version", "2023-06-01")
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("Anthropic health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_embedding_creation() {
        let embedding = AnthropicEmbedding::new(
            "test-key".to_string(),
            "claude-embedding-001".to_string(),
            2048,
        );
        assert_eq!(embedding.dimension(), 2048);
        assert_eq!(embedding.model_name(), "claude-embedding-001");
    }

    #[test]
    fn test_anthropic_chat_creation() {
        let chat = AnthropicChat::new(
            "test-key".to_string(),
            "claude-3-opus-20240229".to_string(),
            0.7,
            2048,
        );
        assert_eq!(chat.model_name(), "claude-3-opus-20240229");
    }
}
