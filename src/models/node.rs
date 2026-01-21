#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use surrealdb::sql::Thing;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::embedding::EmbeddingVector;

/// Tipo de nodo en la estructura fractal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    /// Nodo hoja: contenido original sin procesar
    Leaf,
    /// Nodo padre: resumen/síntesis de nodos hijos
    Parent,
    /// Nodo raíz: nivel más alto del fractal
    Root,
}

/// Estado de completitud del conocimiento
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NodeStatus {
    /// Conocimiento completo y validado
    Complete,
    /// Conocimiento incompleto: requiere investigación en fase REM
    Incomplete,
    /// En proceso de validación/consolidación
    Pending,
    /// Marcado para eliminación o actualización
    Deprecated,
}

/// Nodo fractal: unidad fundamental de conocimiento
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalNode {
    /// Identificador único del nodo (Thing de SurrealDB)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Thing>,

    /// UUID para referencia externa
    pub uuid: Uuid,

    /// Tipo de nodo en la jerarquía fractal
    pub node_type: NodeType,

    /// Estado de completitud
    pub status: NodeStatus,

    /// Contenido textual del nodo
    pub content: String,

    /// Resumen breve (para nodos padre)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Embedding vectorial semántico
    pub embedding: EmbeddingVector,

    /// Nivel en la jerarquía fractal (0 = hoja, mayor = más abstracto)
    pub depth_level: u32,

    /// Confianza semántica (0.0 - 1.0)
    pub confidence: f32,

    /// Namespace del nodo (global o usuario específico)
    pub namespace: String,

    /// Scope del usuario (si es privado)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,

    /// Metadatos adicionales
    pub metadata: NodeMetadata,

    /// Timestamp de creación
    pub created_at: DateTime<Utc>,

    /// Timestamp de última actualización
    pub updated_at: DateTime<Utc>,

    /// Timestamp de última consulta (para cache LRU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_accessed_at: Option<DateTime<Utc>>,
}

/// Metadatos del nodo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Fuente del conocimiento (documento, web, conversación)
    pub source: String,

    /// Tipo de fuente (pdf, image, text, web)
    pub source_type: SourceType,

    /// URL o path del documento original
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,

    /// Autor o creador del contenido
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,

    /// Tags para categorización
    pub tags: Vec<String>,

    /// Lenguaje del contenido (ISO 639-1)
    pub language: String,

    /// Número de veces que se ha accedido (para LRU)
    pub access_count: u64,
}

/// Tipo de fuente de conocimiento
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    /// Documento PDF
    Pdf,
    /// Imagen con OCR
    Image,
    /// Texto plano
    Text,
    /// Página web
    Web,
    /// Conversación/memoria episódica
    Conversation,
    /// Generado por el sistema (resumen)
    Synthetic,
}

impl FractalNode {
    /// Crea un nuevo nodo hoja
    pub fn new_leaf(
        content: String,
        embedding: EmbeddingVector,
        namespace: String,
        scope: Option<String>,
        metadata: NodeMetadata,
    ) -> Self {
        Self {
            id: None,
            uuid: Uuid::new_v4(),
            node_type: NodeType::Leaf,
            status: NodeStatus::Complete,
            content,
            summary: None,
            embedding,
            depth_level: 0,
            confidence: 1.0,
            namespace,
            scope,
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed_at: None,
        }
    }

    /// Crea un nodo padre a partir de nodos hijos
    pub fn new_parent(
        summary: String,
        content: String,
        embedding: EmbeddingVector,
        depth_level: u32,
        namespace: String,
        scope: Option<String>,
        metadata: NodeMetadata,
    ) -> Self {
        Self {
            id: None,
            uuid: Uuid::new_v4(),
            node_type: NodeType::Parent,
            status: NodeStatus::Complete,
            content,
            summary: Some(summary),
            embedding,
            depth_level,
            confidence: 0.85, // Los resúmenes tienen menor confianza inicial
            namespace,
            scope,
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed_at: None,
        }
    }

    /// Marca el nodo como incompleto (requiere fase REM)
    pub fn mark_incomplete(&mut self) {
        self.status = NodeStatus::Incomplete;
        self.updated_at = Utc::now();
    }

    /// Actualiza el timestamp de último acceso
    pub fn touch(&mut self) {
        self.last_accessed_at = Some(Utc::now());
        self.metadata.access_count += 1;
    }

    /// Verifica si el nodo es un candidato para cache LRU
    pub fn is_hot(&self) -> bool {
        self.metadata.access_count > 10
    }

    /// Calcula la relevancia basada en similitud y confianza
    pub fn calculate_relevance(&self, similarity: f32) -> f32 {
        similarity * self.confidence
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            source_type: SourceType::Text,
            source_url: None,
            author: None,
            tags: Vec::new(),
            language: "en".to_string(),
            access_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::embedding::EmbeddingModel;

    #[test]
    fn test_new_leaf_node() {
        let embedding = EmbeddingVector {
            vector: vec![0.1, 0.2, 0.3],
            model: EmbeddingModel::NomicEmbedTextV15,
            dimension: 3,
        };

        let node = FractalNode::new_leaf(
            "Test content".to_string(),
            embedding,
            "global_knowledge".to_string(),
            None,
            NodeMetadata::default(),
        );

        assert_eq!(node.node_type, NodeType::Leaf);
        assert_eq!(node.status, NodeStatus::Complete);
        assert_eq!(node.depth_level, 0);
        assert_eq!(node.confidence, 1.0);
    }

    #[test]
    fn test_mark_incomplete() {
        let embedding = EmbeddingVector {
            vector: vec![0.1, 0.2, 0.3],
            model: EmbeddingModel::NomicEmbedTextV15,
            dimension: 3,
        };

        let mut node = FractalNode::new_leaf(
            "Test content".to_string(),
            embedding,
            "global_knowledge".to_string(),
            None,
            NodeMetadata::default(),
        );

        node.mark_incomplete();
        assert_eq!(node.status, NodeStatus::Incomplete);
    }

    #[test]
    fn test_touch_increments_access_count() {
        let embedding = EmbeddingVector {
            vector: vec![0.1, 0.2, 0.3],
            model: EmbeddingModel::NomicEmbedTextV15,
            dimension: 3,
        };

        let mut node = FractalNode::new_leaf(
            "Test content".to_string(),
            embedding,
            "global_knowledge".to_string(),
            None,
            NodeMetadata::default(),
        );

        assert_eq!(node.metadata.access_count, 0);
        node.touch();
        assert_eq!(node.metadata.access_count, 1);
        assert!(node.last_accessed_at.is_some());
    }

    #[test]
    fn test_is_hot() {
        let embedding = EmbeddingVector {
            vector: vec![0.1, 0.2, 0.3],
            model: EmbeddingModel::NomicEmbedTextV15,
            dimension: 3,
        };

        let mut node = FractalNode::new_leaf(
            "Test content".to_string(),
            embedding,
            "global_knowledge".to_string(),
            None,
            NodeMetadata::default(),
        );

        assert!(!node.is_hot());

        node.metadata.access_count = 11;
        assert!(node.is_hot());
    }
}
