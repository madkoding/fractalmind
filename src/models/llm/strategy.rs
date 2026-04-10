//! Estrategias para usar modelos de lenguaje
//!
//! Este módulo define las estrategias para utilizar modelos locales (Fractal)
//! o remotos (Ollama) para generar embeddings, chat y resúmenes.

#![allow(dead_code)]

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::traits_llm::{ChatMessage, ChatResponse, ChatRole, EmbeddingResponse};
use crate::db::connection::DatabaseConnection;
use crate::db::queries::{NodeRepository, EdgeRepository};
use crate::graph::{Sssp, GraphNode};

// ============================================================================
// FractalModelStrategy Configuration
// ============================================================================

/// Configuración para FractalModelStrategy
#[derive(Debug, Clone)]
pub struct FractalModelStrategyConfig {
    /// Namespace por defecto para búsquedas
    pub default_namespace: String,
    /// Número máximo de resultados para recuperar
    pub max_results: usize,
    /// Peso para similitud vectorial (0.0-1.0)
    pub vector_weight: f32,
    /// Peso para proximidad en el grafo (0.0-1.0)
    pub graph_weight: f32,
    /// URL base para Ollama
    pub ollama_base_url: String,
    /// Temperatura para chat
    pub chat_temperature: f32,
    /// Máximo de tokens para chat
    pub chat_max_tokens: u32,
    /// Temperatura para sumarización
    pub summarizer_temperature: f32,
    /// Máximo de tokens para sumarización
    pub summarizer_max_tokens: u32,
}

impl Default for FractalModelStrategyConfig {
    fn default() -> Self {
        Self {
            default_namespace: "global_knowledge".to_string(),
            max_results: 5,
            vector_weight: 0.7,
            graph_weight: 0.3,
            ollama_base_url: "http://localhost:11434".to_string(),
            chat_temperature: 0.7,
            chat_max_tokens: 2048,
            summarizer_temperature: 0.3,
            summarizer_max_tokens: 512,
        }
    }
}

impl FractalModelStrategyConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_namespace(mut self, namespace: &str) -> Self {
        self.default_namespace = namespace.to_string();
        self
    }

    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    pub fn with_weights(mut self, vector: f32, graph: f32) -> Self {
        self.vector_weight = vector;
        self.graph_weight = graph;
        self
    }

    pub fn with_ollama_base_url(mut self, url: &str) -> Self {
        self.ollama_base_url = url.to_string();
        self
    }

    pub fn with_chat_config(mut self, temperature: f32, max_tokens: u32) -> Self {
        self.chat_temperature = temperature;
        self.chat_max_tokens = max_tokens;
        self
    }

    pub fn with_summarizer_config(mut self, temperature: f32, max_tokens: u32) -> Self {
        self.summarizer_temperature = temperature;
        self.summarizer_max_tokens = max_tokens;
        self
    }

    pub fn from_env() -> Self {
        let default_namespace = std::env::var("FRACTAL_DEFAULT_NAMESPACE")
            .unwrap_or_else(|_| "global_knowledge".to_string());
        
        let max_results = std::env::var("FRACTAL_MAX_RESULTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        
        let ollama_base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());

        Self {
            default_namespace,
            max_results,
            ollama_base_url,
            ..Default::default()
        }
    }
}

// ============================================================================
// Model Strategy Trait
// ============================================================================

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

// ============================================================================
// FractalModelStrategy - Implementación con navegación por grafo
// ============================================================================

/// Estrategia que usa modelos fractales almacenados con navegación por grafo
pub struct FractalModelStrategy {
    model_id: String,
    db: Arc<RwLock<DatabaseConnection>>,
    config: FractalModelStrategyConfig,
}

impl FractalModelStrategy {
    pub fn new(model_id: String, db: DatabaseConnection) -> Self {
        Self { 
            model_id,
            db: Arc::new(RwLock::new(db)),
            config: FractalModelStrategyConfig::default(),
        }
    }

    pub fn with_config(model_id: String, db: DatabaseConnection, config: FractalModelStrategyConfig) -> Self {
        Self { 
            model_id,
            db: Arc::new(RwLock::new(db)),
            config,
        }
    }

    /// Navega por el grafo fractal para encontrar contexto relevante
    async fn navigate_fractal_graph(&self, query_embedding: &[f32], namespace: &str, limit: usize) -> Result<Vec<String>> {
        let db = self.db.read().await;
        let node_repo = NodeRepository::new(&db);
        
        let results = node_repo.search_similar(query_embedding, namespace, limit * 2).await?;
        
        if results.is_empty() {
            return Ok(vec![]);
        }

        let mut graph: std::collections::HashMap<String, GraphNode> = std::collections::HashMap::new();
        let mut node_contents: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        
        for (node, _similarity) in &results {
            if let Some(id) = &node.id {
                let id_str = id.to_string();
                let graph_node = GraphNode::new(id_str.clone(), node.namespace.clone());
                graph.insert(id_str.clone(), graph_node);
                node_contents.insert(id_str, node.content.clone());
            }
        }

        let edge_repo = EdgeRepository::new(&db);
        for (node, _) in &results {
            if let Some(id) = &node.id {
                if let Ok(outgoing) = edge_repo.get_outgoing(id).await {
                    for edge in outgoing {
                        let to_id = edge.to.to_string();
                        if let Some(gn) = graph.get_mut(&id.to_string()) {
                            gn.add_edge(to_id, edge.similarity);
                        }
                    }
                }
            }
        }

        if graph.len() > 1 {
            let sssp = Sssp::with_defaults();
            let start_node = results.first()
                .and_then(|(n, _)| n.id.as_ref())
                .map(|id| id.to_string())
                .unwrap_or_default();

            if !start_node.is_empty() {
                let sssp_result = sssp.compute(&graph, &start_node, None);
                
                let mut ranked: Vec<(String, f32)> = node_contents
                    .keys()
                    .filter_map(|id| {
                        let base_sim = results.iter()
                            .find(|(n, _)| n.id.as_ref().map(|i| i.to_string()) == Some(id.clone()))
                            .map(|(_, s)| *s)
                            .unwrap_or(0.5);
                        
                        let graph_score = sssp_result.distances.get(id)
                            .map(|&d| 1.0 / (1.0 + d))
                            .unwrap_or(0.0);
                        
                        // Usar pesos configurables
                        let combined = base_sim * self.config.vector_weight + graph_score * self.config.graph_weight;
                        Some((id.clone(), combined))
                    })
                    .collect();

                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                return Ok(ranked
                    .into_iter()
                    .take(limit)
                    .filter_map(|(id, _)| node_contents.get(&id).cloned())
                    .collect());
            }
        }

        Ok(results
            .into_iter()
            .take(limit)
            .map(|(n, _)| n.content)
            .collect())
    }

    fn get_default_namespace(&self) -> &str {
        &self.config.default_namespace
    }

    async fn generate_summary_with_context(&self, text: &str) -> Result<String> {
        use super::providers::OllamaSummarizer;
        use super::traits_llm::SummarizerProvider;
        
        let provider = OllamaSummarizer::new(
            self.config.ollama_base_url.clone(),
            self.model_id.clone(),
            self.config.summarizer_temperature,
            self.config.summarizer_max_tokens,
        );
        
        provider.summarize(text).await
    }
}

#[async_trait]
impl ModelStrategy for FractalModelStrategy {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResponse>> {
        use crate::embeddings::EmbeddingService;
        use crate::embeddings::config::EmbeddingConfig;
        use crate::models::EmbeddingModel;
        
        let config = EmbeddingConfig::with_model(EmbeddingModel::NomicEmbedTextV15);
        let service = EmbeddingService::with_mock(config);
        let results = service.embed_batch(&texts).await?;
        
        Ok(results.embeddings.into_iter().map(|emb| EmbeddingResponse {
            embedding: emb.vector,
            dimension: emb.dimension,
            model: self.model_id.clone(),
            latency_ms: 0,
        }).collect())
    }

    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse> {
        let query = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Generar embedding para la query
        let query_embeddings = self.embed_batch(vec![query.clone()]).await?;
        
        // Navegar el grafo para obtener contexto
        let context = if let Some(embedding) = query_embeddings.first() {
            self.navigate_fractal_graph(
                &embedding.embedding,
                self.get_default_namespace(),
                self.config.max_results,
            ).await?
        } else {
            vec![]
        };

        // Construir system prompt con contexto
        let system_prompt = if context.is_empty() {
            "Eres un asistente útil. Responde de manera concisa y precisa.".to_string()
        } else {
            format!(
                "Eres un asistente de conocimiento. Responde basándote ÚNICAMENTE en el siguiente contexto:\n\n{}\n\n---\n\nSi el contexto no tiene suficiente información, indícalo claramente.",
                context.join("\n\n")
            )
        };

        // Construir mensajes con system prompt
        let mut enriched_messages = vec![ChatMessage {
            role: ChatRole::System,
            content: system_prompt,
        }];
        
        enriched_messages.extend(messages);

        // Usar Ollama para generar respuesta
        use super::providers::OllamaChat;
        use super::traits_llm::ChatProvider;
        
        let provider = OllamaChat::new(
            self.config.ollama_base_url.clone(),
            self.model_id.clone(),
            self.config.chat_temperature,
            self.config.chat_max_tokens,
        );
        
        provider.chat(&enriched_messages).await
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        
        
        // Generar embedding para el texto
        let embeddings = self.embed_batch(vec![text.to_string()]).await?;
        
        // Buscar contexto relacionado en el grafo
        if let Some(embedding) = embeddings.first() {
            let context = self.navigate_fractal_graph(
                &embedding.embedding,
                self.get_default_namespace(),
                3,
            ).await?;

            if !context.is_empty() {
                // Enriquecer texto con contexto relacionado
                let enriched_text = format!(
                    "{}\n\nContexto relacionado:\n{}",
                    text,
                    context.join("\n")
                );
                
                return self.generate_summary_with_context(&enriched_text).await;
            }
        }

        // Sin contexto, resumir texto original
        self.generate_summary_with_context(text).await
    }

    fn name(&self) -> &str {
        "FractalModel"
    }
}

// ============================================================================
// OllamaModelStrategy - Implementación directa con Ollama
// ============================================================================

/// Estrategia que usa Ollama directamente
pub struct OllamaModelStrategy {
    base_url: String,
    model_name: String,
    api_key: Option<String>,
    temperature: f32,
    max_tokens: u32,
}

impl OllamaModelStrategy {
    pub fn new(base_url: String, model_name: String) -> Self {
        Self { 
            base_url, 
            model_name, 
            api_key: None,
            temperature: 0.7,
            max_tokens: 2048,
        }
    }

    pub fn with_api_key(base_url: String, model_name: String, api_key: String) -> Self {
        Self { 
            base_url, 
            model_name, 
            api_key: Some(api_key),
            temperature: 0.7,
            max_tokens: 2048,
        }
    }

    pub fn with_config(
        base_url: String, 
        model_name: String,
        temperature: f32,
        max_tokens: u32,
    ) -> Self {
        Self { 
            base_url, 
            model_name, 
            api_key: None,
            temperature,
            max_tokens,
        }
    }
}

#[async_trait]
impl ModelStrategy for OllamaModelStrategy {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResponse>> {
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
        
        let embeddings = provider.embed_batch(&texts).await?;
        
        Ok(embeddings.into_iter().map(|emb| EmbeddingResponse {
            embedding: emb.embedding,
            dimension: emb.dimension,
            model: self.model_name.clone(),
            latency_ms: emb.latency_ms,
        }).collect())
    }

    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse> {
        use super::providers::OllamaChat;
        use super::traits_llm::ChatProvider;
        
        let provider = if let Some(key) = &self.api_key {
            OllamaChat::with_api_key(
                self.base_url.clone(),
                self.model_name.clone(),
                self.temperature,
                self.max_tokens,
                key.clone(),
            )
        } else {
            OllamaChat::new(
                self.base_url.clone(),
                self.model_name.clone(),
                self.temperature,
                self.max_tokens,
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
                0.3,
                512,
                key.clone(),
            )
        } else {
            OllamaSummarizer::new(
                self.base_url.clone(),
                self.model_name.clone(),
                0.3,
                512,
            )
        };
        
        provider.summarize(text).await
    }

    fn name(&self) -> &str {
        "Ollama"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_strategy_config_defaults() {
        let config = FractalModelStrategyConfig::default();
        assert_eq!(config.default_namespace, "global_knowledge");
        assert_eq!(config.max_results, 5);
        assert!((config.vector_weight - 0.7).abs() < f32::EPSILON);
        assert!((config.graph_weight - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fractal_strategy_config_builder() {
        let config = FractalModelStrategyConfig::new()
            .with_namespace("user_alice")
            .with_max_results(10)
            .with_weights(0.8, 0.2);
        
        assert_eq!(config.default_namespace, "user_alice");
        assert_eq!(config.max_results, 10);
        assert!((config.vector_weight - 0.8).abs() < f32::EPSILON);
        assert!((config.graph_weight - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fractal_strategy_creation() {
        let db = crate::db::connection::DatabaseConnection::default();
        let strategy = FractalModelStrategy::new("model:123".to_string(), db);
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

    #[test]
    fn test_ollama_strategy_with_config() {
        let strategy = OllamaModelStrategy::with_config(
            "http://localhost:11434".to_string(),
            "llama2".to_string(),
            0.5,
            1024,
        );
        assert_eq!(strategy.name(), "Ollama");
    }
}
