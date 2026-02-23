#![allow(dead_code)]

use super::config::{BrainConfig, ModelConfig, ModelProvider};
use super::providers::{OllamaChat, OllamaEmbedding, OllamaSummarizer, OpenAIChat, OpenAIEmbedding, AnthropicChat, AnthropicEmbedding};
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

        // Verificar salud de los proveedores
        info!("Verifying provider health...");
        Self::verify_providers_health(
            &embedding_provider,
            &chat_provider,
            &summarizer_provider,
        )
        .await?;

        info!("ModelBrain initialized successfully");

        Ok(Self {
            embedding_provider,
            chat_provider,
            summarizer_provider,
            config,
        })
    }

    /// Crea un nuevo cerebro sin verificar salud de providers (para arranque sin Ollama)
    pub fn new_without_health_check(config: BrainConfig) -> Result<Self> {
        info!("Initializing ModelBrain without health check");

        // Validar configuración
        config.validate()?;

        // Inicializar proveedores sin verificar salud
        let embedding_provider = Self::create_embedding_provider(&config.embedding_model)?;
        let chat_provider = Self::create_chat_provider(&config.chat_model)?;
        let summarizer_provider = Self::create_summarizer_provider(&config.summarizer_model)?;

        info!("ModelBrain initialized (health check skipped)");

        Ok(Self {
            embedding_provider,
            chat_provider,
            summarizer_provider,
            config,
        })
    }

    /// Crea un proveedor de embeddings desde configuración
    fn create_embedding_provider(
        config: &ModelConfig,
    ) -> Result<Arc<dyn EmbeddingProvider>> {
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
            name if name.contains("bge-small") => 384,
            name if name.contains("all-MiniLM") => 384,
            name if name.contains("text-embedding-ada-002") => 1536,
            name if name.contains("text-embedding-3-small") => 1536,
            name if name.contains("text-embedding-3-large") => 3072,
            _ => {
                warn!(
                    "Unknown embedding model dimension for {}, defaulting to 768",
                    model_name
                );
                768
            }
        }
    }

    /// Verifica la salud de todos los proveedores
    async fn verify_providers_health(
        embedding: &Arc<dyn EmbeddingProvider>,
        chat: &Arc<dyn ChatProvider>,
        summarizer: &Arc<dyn SummarizerProvider>,
    ) -> Result<()> {
        info!("Verifying provider health...");

        let embedding_ok = match embedding.health_check().await {
            Ok(ok) => {
                if ok {
                    info!("✅ Embedding provider ({}): healthy", embedding.model_name());
                } else {
                    warn!("❌ Embedding provider ({}): health check failed", embedding.model_name());
                }
                ok
            }
            Err(e) => {
                error!("❌ Embedding provider health check error: {}", e);
                false
            }
        };

        let chat_ok = match chat.health_check().await {
            Ok(ok) => {
                if ok {
                    info!("✅ Chat provider ({}): healthy", chat.model_name());
                } else {
                    warn!("❌ Chat provider ({}): health check failed", chat.model_name());
                }
                ok
            }
            Err(e) => {
                error!("❌ Chat provider health check error: {}", e);
                false
            }
        };

        let summarizer_ok = match summarizer.health_check().await {
            Ok(ok) => {
                if ok {
                    info!("✅ Summarizer provider ({}): healthy", summarizer.model_name());
                } else {
                    warn!("❌ Summarizer provider ({}): health check failed", summarizer.model_name());
                }
                ok
            }
            Err(e) => {
                error!("❌ Summarizer provider health check error: {}", e);
                false
            }
        };

        if !embedding_ok || !chat_ok || !summarizer_ok {
            error!(
                "❌ Provider health check failed - System cannot operate without working providers"
            );
            error!("embedding: {}, chat: {}, summarizer: {}", 
                   if embedding_ok { "ok" } else { "failed" },
                   if chat_ok { "ok" } else { "failed" },
                   if summarizer_ok { "ok" } else { "failed" });
            
            let mut error_msg = String::from("Provider configuration error:\n");
            if !embedding_ok {
                error_msg.push_str(&format!("- Embedding provider '{}': failed health check\n", embedding.model_name()));
                error_msg.push_str("  → Ensure the provider is running and accessible\n");
                error_msg.push_str("  → Check your network connection and provider URL\n");
                error_msg.push_str("  → Verify your API key if using cloud service\n");
            }
            if !chat_ok {
                error_msg.push_str(&format!("- Chat provider '{}': failed health check\n", chat.model_name()));
                error_msg.push_str("  → Ensure the provider is running and accessible\n");
                error_msg.push_str("  → Check your network connection and provider URL\n");
                error_msg.push_str("  → Verify your API key if using cloud service\n");
            }
            if !summarizer_ok {
                error_msg.push_str(&format!("- Summarizer provider '{}': failed health check\n", summarizer.model_name()));
                error_msg.push_str("  → Ensure the provider is running and accessible\n");
                error_msg.push_str("  → Check your network connection and provider URL\n");
                error_msg.push_str("  → Verify your API key if using cloud service\n");
            }
            
            return Err(anyhow::anyhow!(error_msg.trim().to_string()));
        }

        info!("✅ All providers are healthy and ready");
        Ok(())
    }

    /// Genera un embedding para texto
    pub async fn embed(&self, text: &str) -> Result<EmbeddingResponse> {
        self.embedding_provider.embed(text).await
    }

    /// Genera embeddings en batch
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResponse>> {
        self.embedding_provider.embed_batch(texts).await
    }

    /// Genera una respuesta de chat
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse> {
        self.chat_provider.chat(messages).await
    }

    /// Genera una respuesta simple de chat
    pub async fn simple_chat(&self, user_message: &str) -> Result<ChatResponse> {
        self.chat_provider.simple_chat(user_message).await
    }

    /// Genera una respuesta con contexto del sistema
    pub async fn chat_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> Result<ChatResponse> {
        self.chat_provider
            .chat_with_system(system_prompt, user_message)
            .await
    }

    /// Resume un texto (fase REM)
    pub async fn summarize(&self, text: &str) -> Result<String> {
        self.summarizer_provider.summarize(text).await
    }

    /// Resume múltiples textos y combina en resumen jerárquico (RAPTOR)
    pub async fn summarize_batch(&self, texts: &[String]) -> Result<String> {
        self.summarizer_provider.summarize_batch(texts).await
    }

    /// Resume con prompt personalizado
    pub async fn summarize_with_prompt(&self, text: &str, custom_prompt: &str) -> Result<String> {
        self.summarizer_provider
            .summarize_with_prompt(text, custom_prompt)
            .await
    }

    /// Obtiene información sobre los modelos activos
    pub fn get_models_info(&self) -> ModelsInfo {
        ModelsInfo {
            embedding_model: self.embedding_provider.model_name().to_string(),
            embedding_dimension: self.embedding_provider.dimension(),
            chat_model: self.chat_provider.model_name().to_string(),
            summarizer_model: self.summarizer_provider.model_name().to_string(),
            prefer_local: self.config.prefer_local,
        }
    }

    /// Verifica si todos los proveedores están usando modelos locales
    pub fn is_fully_local(&self) -> bool {
        self.config.embedding_model.provider.is_local()
            && self.config.chat_model.provider.is_local()
            && self.config.summarizer_model.provider.is_local()
    }
}

/// Información sobre los modelos activos
#[derive(Debug, Clone)]
pub struct ModelsInfo {
    pub embedding_model: String,
    pub embedding_dimension: usize,
    pub chat_model: String,
    pub summarizer_model: String,
    pub prefer_local: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_embedding_dimension() {
        assert_eq!(ModelBrain::infer_embedding_dimension("nomic-embed-text"), 768);
        assert_eq!(
            ModelBrain::infer_embedding_dimension("bge-small-en-v1.5"),
            384
        );
        assert_eq!(
            ModelBrain::infer_embedding_dimension("text-embedding-ada-002"),
            1536
        );
        assert_eq!(
            ModelBrain::infer_embedding_dimension("text-embedding-3-large"),
            3072
        );
    }

    #[tokio::test]
    #[ignore] // Requiere Ollama corriendo
    async fn test_create_model_brain_local() {
        let config = BrainConfig::default_local();
        let brain = ModelBrain::new(config).await;
        assert!(brain.is_ok());

        let brain = brain.unwrap();
        assert!(brain.is_fully_local());

        let info = brain.get_models_info();
        assert_eq!(info.embedding_dimension, 768);
    }
}
