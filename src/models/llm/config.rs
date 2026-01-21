#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::{Result, bail};

/// Tipo de modelo en el sistema
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Modelo para generar embeddings
    Embedding,
    /// Modelo para chat/respuestas finales
    Chat,
    /// Modelo para resumir (fase REM/RAPTOR)
    Summarizer,
}

/// Proveedor de modelo (local o remoto)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    /// Ollama local (soberanía de datos)
    Ollama {
        base_url: String,
        model_name: String,
    },
    /// OpenAI API (remoto)
    OpenAI {
        api_key: String,
        model_name: String,
        organization: Option<String>,
    },
    /// Anthropic Claude API (remoto)
    Anthropic {
        api_key: String,
        model_name: String,
    },
    /// Hugging Face API (remoto)
    HuggingFace {
        api_key: String,
        model_name: String,
    },
    /// Candle local (Rust inference)
    Candle {
        model_path: String,
        model_type: String,
    },
    /// Proveedor personalizado
    Custom {
        name: String,
        endpoint: String,
        headers: HashMap<String, String>,
    },
}

impl ModelProvider {
    /// Verifica si el proveedor es local (soberanía de datos)
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            ModelProvider::Ollama { .. } | ModelProvider::Candle { .. }
        )
    }

    /// Verifica si el proveedor requiere API key
    pub fn requires_api_key(&self) -> bool {
        matches!(
            self,
            ModelProvider::OpenAI { .. }
                | ModelProvider::Anthropic { .. }
                | ModelProvider::HuggingFace { .. }
        )
    }

    /// Obtiene el nombre del modelo
    pub fn model_name(&self) -> String {
        match self {
            ModelProvider::Ollama { model_name, .. } => model_name.clone(),
            ModelProvider::OpenAI { model_name, .. } => model_name.clone(),
            ModelProvider::Anthropic { model_name, .. } => model_name.clone(),
            ModelProvider::HuggingFace { model_name, .. } => model_name.clone(),
            ModelProvider::Candle { model_path, .. } => model_path.clone(),
            ModelProvider::Custom { name, .. } => name.clone(),
        }
    }
}

/// Configuración de un modelo específico
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Tipo de modelo
    pub model_type: ModelType,

    /// Proveedor del modelo
    pub provider: ModelProvider,

    /// Temperatura (creatividad) - 0.0 a 2.0
    pub temperature: f32,

    /// Top-p (nucleus sampling) - 0.0 a 1.0
    pub top_p: f32,

    /// Máximo de tokens en la respuesta
    pub max_tokens: u32,

    /// Timeout en segundos
    pub timeout_seconds: u64,

    /// Reintentos en caso de fallo
    pub max_retries: u32,

    /// Configuración adicional específica del proveedor
    pub extra_config: HashMap<String, String>,
}

impl ModelConfig {
    /// Crea una configuración por defecto para embeddings con Ollama
    pub fn default_embedding_ollama() -> Self {
        Self {
            model_type: ModelType::Embedding,
            provider: ModelProvider::Ollama {
                base_url: "http://localhost:11434".to_string(),
                model_name: "nomic-embed-text".to_string(),
            },
            temperature: 0.0, // Embeddings no usan temperatura
            top_p: 1.0,
            max_tokens: 0, // Embeddings no generan tokens
            timeout_seconds: 30,
            max_retries: 3,
            extra_config: HashMap::new(),
        }
    }

    /// Crea una configuración por defecto para chat con Ollama
    pub fn default_chat_ollama() -> Self {
        Self {
            model_type: ModelType::Chat,
            provider: ModelProvider::Ollama {
                base_url: "http://localhost:11434".to_string(),
                model_name: "llama2".to_string(),
            },
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 2048,
            timeout_seconds: 60,
            max_retries: 2,
            extra_config: HashMap::new(),
        }
    }

    /// Crea una configuración por defecto para summarizer con Ollama
    pub fn default_summarizer_ollama() -> Self {
        Self {
            model_type: ModelType::Summarizer,
            provider: ModelProvider::Ollama {
                base_url: "http://localhost:11434".to_string(),
                model_name: "llama2".to_string(),
            },
            temperature: 0.3, // Más determinista para resúmenes consistentes
            top_p: 0.9,
            max_tokens: 512, // Resúmenes más cortos
            timeout_seconds: 45,
            max_retries: 2,
            extra_config: HashMap::new(),
        }
    }

    /// Crea una configuración para OpenAI
    pub fn openai_chat(api_key: String, model: &str) -> Self {
        Self {
            model_type: ModelType::Chat,
            provider: ModelProvider::OpenAI {
                api_key,
                model_name: model.to_string(),
                organization: None,
            },
            temperature: 0.7,
            top_p: 1.0,
            max_tokens: 2048,
            timeout_seconds: 60,
            max_retries: 3,
            extra_config: HashMap::new(),
        }
    }

    /// Verifica si la configuración es válida
    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 || self.temperature > 2.0 {
            bail!("Temperature must be between 0.0 and 2.0");
        }

        if self.top_p < 0.0 || self.top_p > 1.0 {
            bail!("Top-p must be between 0.0 and 1.0");
        }

        if self.provider.requires_api_key() {
            match &self.provider {
                ModelProvider::OpenAI { api_key, .. }
                | ModelProvider::Anthropic { api_key, .. }
                | ModelProvider::HuggingFace { api_key, .. } => {
                    if api_key.is_empty() {
                        bail!("API key is required for remote provider");
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Configuración completa del "cerebro" (todos los modelos)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    /// Modelo para embeddings
    pub embedding_model: ModelConfig,

    /// Modelo para chat
    pub chat_model: ModelConfig,

    /// Modelo para summarización (fase REM)
    pub summarizer_model: ModelConfig,

    /// Priorizar modelos locales para soberanía de datos
    pub prefer_local: bool,
}

impl BrainConfig {
    /// Crea una configuración por defecto (todo local con Ollama)
    pub fn default_local() -> Self {
        Self {
            embedding_model: ModelConfig::default_embedding_ollama(),
            chat_model: ModelConfig::default_chat_ollama(),
            summarizer_model: ModelConfig::default_summarizer_ollama(),
            prefer_local: true,
        }
    }

    /// Crea una configuración híbrida (embeddings local, chat remoto)
    pub fn hybrid(openai_api_key: String) -> Self {
        Self {
            embedding_model: ModelConfig::default_embedding_ollama(),
            chat_model: ModelConfig::openai_chat(openai_api_key.clone(), "gpt-4"),
            summarizer_model: ModelConfig::default_summarizer_ollama(),
            prefer_local: false,
        }
    }

    /// Valida toda la configuración
    pub fn validate(&self) -> Result<()> {
        self.embedding_model.validate()?;
        self.chat_model.validate()?;
        self.summarizer_model.validate()?;
        Ok(())
    }

    /// Carga configuración desde variables de entorno
    pub fn from_env() -> Result<Self> {
        let prefer_local = std::env::var("LLM_PREFER_LOCAL")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);

        let ollama_base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());

        // Embedding model
        let embedding_provider = std::env::var("EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| "ollama".to_string());
        let embedding_model_name = std::env::var("EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());
        let embedding_model = match embedding_provider.as_str() {
            "ollama" => ModelConfig {
                model_type: ModelType::Embedding,
                provider: ModelProvider::Ollama {
                    base_url: ollama_base_url.clone(),
                    model_name: embedding_model_name,
                },
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 0,
                timeout_seconds: 30,
                max_retries: 3,
                extra_config: HashMap::new(),
            },
            _ => bail!("Unsupported embedding provider: {}", embedding_provider),
        };

        // Chat model
        let chat_provider = std::env::var("CHAT_PROVIDER")
            .unwrap_or_else(|_| "ollama".to_string());
        let chat_model_name = std::env::var("CHAT_MODEL")
            .unwrap_or_else(|_| "llama2".to_string());
        let chat_temperature: f32 = std::env::var("CHAT_TEMPERATURE")
            .unwrap_or_else(|_| "0.7".to_string())
            .parse()
            .unwrap_or(0.7);
        let chat_max_tokens: u32 = std::env::var("CHAT_MAX_TOKENS")
            .unwrap_or_else(|_| "2048".to_string())
            .parse()
            .unwrap_or(2048);
        let chat_model = match chat_provider.as_str() {
            "ollama" => ModelConfig {
                model_type: ModelType::Chat,
                provider: ModelProvider::Ollama {
                    base_url: ollama_base_url.clone(),
                    model_name: chat_model_name,
                },
                temperature: chat_temperature,
                top_p: 0.9,
                max_tokens: chat_max_tokens,
                timeout_seconds: 60,
                max_retries: 2,
                extra_config: HashMap::new(),
            },
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY required for OpenAI provider"))?;
                let model = std::env::var("OPENAI_MODEL")
                    .unwrap_or_else(|_| "gpt-4".to_string());
                ModelConfig::openai_chat(api_key, &model)
            }
            _ => bail!("Unsupported chat provider: {}", chat_provider),
        };

        // Summarizer model
        let summarizer_provider = std::env::var("SUMMARIZER_PROVIDER")
            .unwrap_or_else(|_| "ollama".to_string());
        let summarizer_model_name = std::env::var("SUMMARIZER_MODEL")
            .unwrap_or_else(|_| "llama2".to_string());
        let summarizer_temperature: f32 = std::env::var("SUMMARIZER_TEMPERATURE")
            .unwrap_or_else(|_| "0.3".to_string())
            .parse()
            .unwrap_or(0.3);
        let summarizer_max_tokens: u32 = std::env::var("SUMMARIZER_MAX_TOKENS")
            .unwrap_or_else(|_| "512".to_string())
            .parse()
            .unwrap_or(512);
        let summarizer_model = match summarizer_provider.as_str() {
            "ollama" => ModelConfig {
                model_type: ModelType::Summarizer,
                provider: ModelProvider::Ollama {
                    base_url: ollama_base_url,
                    model_name: summarizer_model_name,
                },
                temperature: summarizer_temperature,
                top_p: 0.9,
                max_tokens: summarizer_max_tokens,
                timeout_seconds: 45,
                max_retries: 2,
                extra_config: HashMap::new(),
            },
            _ => bail!("Unsupported summarizer provider: {}", summarizer_provider),
        };

        let config = Self {
            embedding_model,
            chat_model,
            summarizer_model,
            prefer_local,
        };

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_provider_is_local() {
        let ollama = ModelProvider::Ollama {
            base_url: "http://localhost:11434".to_string(),
            model_name: "llama2".to_string(),
        };
        assert!(ollama.is_local());

        let openai = ModelProvider::OpenAI {
            api_key: "test".to_string(),
            model_name: "gpt-4".to_string(),
            organization: None,
        };
        assert!(!openai.is_local());
    }

    #[test]
    fn test_model_config_validation() {
        let mut config = ModelConfig::default_chat_ollama();
        assert!(config.validate().is_ok());

        config.temperature = 3.0; // Invalid
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_brain_config_default_local() {
        let config = BrainConfig::default_local();
        assert!(config.prefer_local);
        assert!(config.embedding_model.provider.is_local());
        assert!(config.chat_model.provider.is_local());
        assert!(config.summarizer_model.provider.is_local());
    }

    #[test]
    fn test_brain_config_validation() {
        let config = BrainConfig::default_local();
        assert!(config.validate().is_ok());
    }
}
