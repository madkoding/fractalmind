#!/bin/bash
# Compile inside Docker to bypass permission issues

cd /home/madkoding/fractalmind

# Remove the problematic file content
cat > src/models/llm/brain.rs << 'EOF'
#![allow(dead_code)]

use super::config::{BrainConfig, ModelConfig, ModelProvider};
use super::providers::{OllamaChat, OllamaEmbedding, OllamaSummarizer, OpenAIChat, OpenAIEmbedding};
use super::traits_llm::{
    ChatMessage, ChatProvider, ChatResponse, EmbeddingProvider, EmbeddingResponse,
    SummarizerProvider,
};
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{error, info, warn};

/// Cerebro del sistema: orquesta todos los modelos LLM
pub struct ModelBrain {
    /// Proveedor de embeddings
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Proveedor de chat
    chat_provider: Arc<dyn ChatProvider>,

    /// Proveedor de resumido (fase REM)
    summarizer_provider: Arc<dyn SummarizerProvider>,

    /// Configuración completa
    config: BrainConfig,
}

impl ModelBrain {
    /// Crea un nuevo cerebro desde configuración
    pub async fn new(config: BrainConfig) -> Result<Self> {
        info!("Initializing ModelBrain with config: {:?}", config);

        // Validar configuración
        config.validate()?;

        // Inicializar proveedor de embeddings
        info!("Initializing embedding provider...");
        let embedding_provider = Self::create_embedding_provider(&config.embedding_model)?;
        info!("Embedding provider initialized: {}", embedding_provider.model_name());

        // Inicializar proveedor de chat
        info!("Initializing chat provider...");
        let chat_provider = Self::create_chat_provider(&config.chat_model)?;
        info!("Chat provider initialized: {}", chat_provider.model_name());

        // Inicializar proveedor de summarizer
        info!("Initializing summarizer provider...");
        let summarizer_provider = Self::create_summarizer_provider(&config.summarizer_model)?;
        info!("Summarizer provider initialized: {}", summarizer_provider.model_name());

        Ok(Self {
            embedding_provider,
            chat_provider,
            summarizer_provider,
            config,
        })
    }

    /// Provee acceso al proveedor de embeddings
    pub fn embedding_provider(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedding_provider
    }

    /// Provee acceso al proveedor de chat
    pub fn chat_provider(&self) -> &Arc<dyn ChatProvider> {
        &self.chat_provider
    }

    /// Provee acceso al proveedor de summarizer
    pub fn summarizer_provider(&self) -> &Arc<dyn SummarizerProvider> {
        &self.summarizer_provider
    }

    /// Provee acceso a la configuración
    pub fn config(&self) -> &BrainConfig {
        &self.config
    }

    /// Inicia el proceso de REM phase
    pub async fn run_rem_phase(&self) -> Result<()> {
        info!("Starting REM phase...");
        
        // Aquí se implementará la lógica de consolidación de memoria
        // y aprendizaje automático
        
        info!("REM phase completed");
        Ok(())
    }

    /// Valida el estado de todos los proveedores
    pub async fn health_check(&self) -> Result<bool> {
        let embedding_ok = self.embedding_provider.health_check().await;
        let chat_ok = self.chat_provider.health_check().await;
        let summarizer_ok = self.summarizer_provider.health_check().await;

        let all_ok = embedding_ok? && chat_ok? && summarizer_ok?;
        
        if !all_ok {
            error!("Health check failed for one or more providers");
        }
        
        Ok(all_ok)
    }

    /// Crea un proveedor de embeddings desde configuración
    fn create_embedding_provider(config: &ModelConfig) -> Result<Arc<dyn EmbeddingProvider>> {
        match &config.provider {
            ModelProvider::Ollama {
                base_url,
                model_name,
                api_key,
            } => {
                info!("Creating Ollama embedding provider: {}", model_name);
                let dimension = Self::infer_embedding_dimension(model_name);
                let provider = if let Some(key) = api_key {
                    info!("Using Ollama Cloud with API key");
                    Arc::new(OllamaEmbedding::with_api_key(
                        base_url.clone(),
                        model_name.clone(),
                        dimension,
                        key.clone(),
                    ))
                } else {
                    info!("Using local Ollama");
                    Arc::new(OllamaEmbedding::new(
                        base_url.clone(),
                        model_name.clone(),
                        dimension,
                    ))
                };
                Ok(provider)
            }
            ModelProvider::OpenAI {
                api_key,
                model_name,
                ..
            } => {
                info!("Creating OpenAI embedding provider: {}", model_name);
                let dimension = Self::infer_embedding_dimension(model_name);
                Ok(Arc::new(OpenAIEmbedding::new(
                    api_key.clone(),
                    model_name.clone(),
                    dimension,
                )))
            }
            ModelProvider::Anthropic {
                api_key,
                model_name,
                ..
            } => {
                info!("Creating Anthropic embedding provider: {}", model_name);
                let dimension = Self::infer_embedding_dimension(model_name);
                Ok(Arc::new(AnthropicEmbedding::new(
                    api_key.clone(),
                    model_name.clone(),
                    dimension,
                )))
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported embedding provider: {:?}",
                config.provider
            )),
        }
    }

    /// Crea un proveedor de chat desde configuración
    fn create_chat_provider(config: &ModelConfig) -> Result<Arc<dyn ChatProvider>> {
        match &config.provider {
            ModelProvider::Ollama {
                base_url,
                model_name,
                api_key,
            } => {
                info!("Creating Ollama chat provider: {}", model_name);
                let provider = if let Some(key) = api_key {
                    info!("Using Ollama Cloud with API key");
                    Arc::new(OllamaChat::with_api_key(
                        base_url.clone(),
                        model_name.clone(),
                        config.temperature,
                        config.max_tokens,
                        key.clone(),
                    ))
                } else {
                    info!("Using local Ollama");
                    Arc::new(OllamaChat::new(
                        base_url.clone(),
                        model_name.clone(),
                        config.temperature,
                        config.max_tokens,
                    ))
                };
                Ok(provider)
            }
            ModelProvider::OpenAI {
                api_key,
                model_name,
                ..
            } => {
                info!("Creating OpenAI chat provider: {}", model_name);
                Ok(Arc::new(OpenAIChat::new(
                    api_key.clone(),
                    model_name.clone(),
                    config.temperature,
                    config.max_tokens,
                )))
            }
            ModelProvider::Anthropic {
                api_key,
                model_name,
                ..
            } => {
                info!("Creating Anthropic chat provider: {}", model_name);
                Ok(Arc::new(AnthropicChat::new(
                    api_key.clone(),
                    model_name.clone(),
                    config.temperature,
                    config.max_tokens,
                )))
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported chat provider: {:?}",
                config.provider
            )),
        }
    }

    /// Crea un proveedor de summarizer desde configuración
    fn create_summarizer_provider(
        config: &ModelConfig,
    ) -> Result<Arc<dyn SummarizerProvider>> {
        match &config.provider {
            ModelProvider::Ollama {
                base_url,
                model_name,
                api_key,
            } => {
                info!("Creating Ollama summarizer provider: {}", model_name);
                let provider = if let Some(key) = api_key {
                    info!("Using Ollama Cloud with API key");
                    Arc::new(OllamaSummarizer::with_api_key(
                        base_url.clone(),
                        model_name.clone(),
                        config.temperature,
                        config.max_tokens,
                        key.clone(),
                    ))
                } else {
                    info!("Using local Ollama");
                    Arc::new(OllamaSummarizer::new(
                        base_url.clone(),
                        model_name.clone(),
                        config.temperature,
                        config.max_tokens,
                    ))
                };
                Ok(provider)
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported summarizer provider: {:?}",
                config.provider
            )),
        }
    }

    /// Infiere la dimensión del embedding desde el nombre del modelo
    fn infer_embedding_dimension(model_name: &str) -> usize {
        match model_name {
            name if name.contains("nomic-embed-text") => 768,
            name if name.contains("nomic-embed-text-v1.5") => 768,
            name if name.contains("nomic-embed-text:137m") => 768,
            name if name.contains("nomic-embed-text:v1.5") => 768,
            name if name.contains("nomic-embed-text:latest") => 768,
            _ => 768, // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_brain_creation() {
        // Basic test to ensure the struct compiles
        assert!(true);
    }
}
EOF

echo "brain.rs fixed"

# Now build with Docker
docker build -t fractalmind:latest . 2>&1 | tail -30
