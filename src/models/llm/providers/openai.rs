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

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Cliente OpenAI para embeddings
pub struct OpenAIEmbedding {
    client: Client,
    api_key: String,
    model_name: String,
    dimension: usize,
}

impl OpenAIEmbedding {
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
struct OpenAIEmbedRequest {
    input: String,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedData>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbedData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    total_tokens: u32,
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbedding {
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse> {
        let start = Instant::now();
        let url = format!("{}/embeddings", OPENAI_API_BASE);

        let request = OpenAIEmbedRequest {
            input: text.to_string(),
            model: self.model_name.clone(),
        };

        debug!("Sending embedding request to OpenAI");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send embedding request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI embedding request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let openai_response: OpenAIEmbedResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI embedding response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let embedding = openai_response
            .data
            .first()
            .context("No embedding data in OpenAI response")?
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
        // OpenAI soporta batch embeddings nativamente
        let start = Instant::now();
        let url = format!("{}/embeddings", OPENAI_API_BASE);

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
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send batch embedding request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI batch embedding request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let openai_response: OpenAIEmbedResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI batch embedding response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let results = openai_response
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
        let url = format!("{}/models", OPENAI_API_BASE);
        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("OpenAI health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

/// Cliente OpenAI para chat
pub struct OpenAIChat {
    client: Client,
    api_key: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
}

impl OpenAIChat {
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
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAIChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChatChoice>,
    usage: OpenAIChatUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatChoice {
    message: OpenAIChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatUsage {
    total_tokens: u32,
}

#[async_trait]
impl ChatProvider for OpenAIChat {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse> {
        let start = Instant::now();
        let url = format!("{}/chat/completions", OPENAI_API_BASE);

        let openai_messages: Vec<OpenAIChatMessage> = messages
            .iter()
            .map(|msg| OpenAIChatMessage {
                role: match msg.role {
                    ChatRole::System => "system".to_string(),
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                },
                content: msg.content.clone(),
            })
            .collect();

        let request = OpenAIChatRequest {
            model: self.model_name.clone(),
            messages: openai_messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        };

        debug!("Sending chat request to OpenAI");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send chat request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI chat request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let openai_response: OpenAIChatResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI chat response")?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let content = openai_response
            .choices
            .first()
            .context("No choices in OpenAI response")?
            .message
            .content
            .clone();

        Ok(ChatResponse {
            content,
            tokens_used: Some(openai_response.usage.total_tokens),
            latency_ms,
            model: self.model_name.clone(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", OPENAI_API_BASE);
        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                warn!("OpenAI health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embedding_creation() {
        let embedding = OpenAIEmbedding::new(
            "test-key".to_string(),
            "text-embedding-ada-002".to_string(),
            1536,
        );
        assert_eq!(embedding.dimension(), 1536);
        assert_eq!(embedding.model_name(), "text-embedding-ada-002");
    }

    #[test]
    fn test_openai_chat_creation() {
        let chat = OpenAIChat::new("test-key".to_string(), "gpt-4".to_string(), 0.7, 2048);
        assert_eq!(chat.model_name(), "gpt-4");
    }
}
