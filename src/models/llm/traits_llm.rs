#![allow(dead_code)]

use async_trait::async_trait;
use anyhow::Result;

/// Mensaje de chat
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Rol del mensaje
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Respuesta de chat
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Contenido de la respuesta
    pub content: String,

    /// Tokens usados (si disponible)
    pub tokens_used: Option<u32>,

    /// Tiempo de respuesta en milisegundos
    pub latency_ms: u64,

    /// Modelo que generó la respuesta
    pub model: String,
}

/// Respuesta de embedding
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// Vector de embedding
    pub embedding: Vec<f32>,

    /// Dimensión del vector
    pub dimension: usize,

    /// Modelo que generó el embedding
    pub model: String,

    /// Tiempo de respuesta en milisegundos
    pub latency_ms: u64,
}

/// Trait para proveedores de embeddings
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Genera un embedding para un texto
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse>;

    /// Genera embeddings para múltiples textos (batch)
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResponse>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Obtiene la dimensión del embedding
    fn dimension(&self) -> usize;

    /// Nombre del modelo
    fn model_name(&self) -> &str;

    /// Verifica si el proveedor está disponible
    async fn health_check(&self) -> Result<bool>;
}

/// Trait para proveedores de chat
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Genera una respuesta de chat
    async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse>;

    /// Genera una respuesta simple (un solo mensaje de usuario)
    async fn simple_chat(&self, user_message: &str) -> Result<ChatResponse> {
        let messages = vec![ChatMessage::user(user_message)];
        self.chat(&messages).await
    }

    /// Genera una respuesta con contexto del sistema
    async fn chat_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> Result<ChatResponse> {
        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_message),
        ];
        self.chat(&messages).await
    }

    /// Nombre del modelo
    fn model_name(&self) -> &str;

    /// Verifica si el proveedor está disponible
    async fn health_check(&self) -> Result<bool>;
}

/// Trait para proveedores de resumido (fase REM / RAPTOR)
#[async_trait]
pub trait SummarizerProvider: Send + Sync {
    /// Resume un texto largo en un resumen corto
    async fn summarize(&self, text: &str) -> Result<String>;

    /// Resume múltiples textos y combina en un resumen jerárquico
    async fn summarize_batch(&self, texts: &[String]) -> Result<String> {
        // Por defecto, resumir cada texto y luego resumir los resúmenes
        let mut summaries = Vec::with_capacity(texts.len());
        for text in texts {
            summaries.push(self.summarize(text).await?);
        }

        // Combinar todos los resúmenes en uno solo
        let combined = summaries.join("\n\n");
        self.summarize(&combined).await
    }

    /// Resume con un prompt personalizado
    async fn summarize_with_prompt(&self, text: &str, custom_prompt: &str) -> Result<String>;

    /// Nombre del modelo
    fn model_name(&self) -> &str;

    /// Verifica si el proveedor está disponible
    async fn health_check(&self) -> Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let system = ChatMessage::system("You are a helpful assistant");
        assert_eq!(system.role, ChatRole::System);

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, ChatRole::User);

        let assistant = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant.role, ChatRole::Assistant);
    }
}
