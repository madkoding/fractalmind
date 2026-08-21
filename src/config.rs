//! Secrets configuration loaded from config/secrets.json
//! This file is NEVER committed to git.

use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

const DEFAULT_SECRETS_PATH: &str = "config/secrets.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecretsConfig {
    pub ollama_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub tavily_api_key: Option<String>,
    pub serper_api_key: Option<String>,
}

impl SecretsConfig {
    pub fn load() -> Self {
        Self::load_from_path(DEFAULT_SECRETS_PATH)
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();

        if !path.exists() {
            info!(
                "No secrets.json found at {}, using environment variables",
                path.display()
            );
            return Self::default();
        }

        match std::fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<SecretsConfig>(&content) {
                Ok(config) => {
                    info!("Loaded secrets from {}", path.display());
                    config
                }
                Err(e) => {
                    warn!("Failed to parse secrets.json: {}. Using defaults.", e);
                    Self::default()
                }
            },
            Err(e) => {
                warn!("Failed to read secrets.json: {}. Using defaults.", e);
                Self::default()
            }
        }
    }

    pub fn get_ollama_api_key(&self) -> Option<String> {
        self.ollama_api_key.clone()
    }
}

pub fn load_secrets() -> SecretsConfig {
    SecretsConfig::load()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_nonexistent_file() {
        let config = SecretsConfig::load_from_path("/nonexistent/path/secrets.json");
        assert!(config.ollama_api_key.is_none());
    }

    #[test]
    fn test_load_valid_secrets() {
        let temp_dir = TempDir::new().unwrap();
        let secrets_path = temp_dir.path().join("secrets.json");

        std::fs::write(&secrets_path, r#"{"ollama_api_key": "sk-test123"}"#).unwrap();

        let config = SecretsConfig::load_from_path(&secrets_path);
        assert_eq!(config.ollama_api_key, Some("sk-test123".to_string()));
    }
}
