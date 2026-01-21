#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Estado del modelo fractal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FractalModelStatus {
    /// Subiendo archivo
    Uploading,
    /// Convirtiendo a estructura fractal
    Converting,
    /// Listo para uso
    Ready,
    /// Falló la conversión
    Failed,
}

/// Arquitectura del modelo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Tipo de arquitectura (llama, mistral, phi, etc)
    pub model_type: String,
    /// Número de capas
    pub n_layers: u32,
    /// Dimensión de embeddings
    pub embedding_dim: u32,
    /// Tamaño del vocabulario
    pub vocab_size: u32,
    /// Número de cabezas de atención
    pub n_heads: u32,
    /// Dimensión de feed-forward
    pub ffn_dim: u32,
}

/// Modelo fractal almacenado en SurrealDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalModel {
    /// ID del modelo
    pub id: String,
    /// Nombre del modelo
    pub name: String,
    /// Arquitectura del modelo
    pub architecture: ModelArchitecture,
    /// ID del nodo raíz del grafo fractal
    pub root_node_id: Option<String>,
    /// Estado actual
    pub status: FractalModelStatus,
    /// Ruta del archivo GGUF original
    pub file_path: String,
    /// Tamaño del archivo en bytes
    pub file_size: u64,
    /// Progreso de conversión (0-100)
    pub conversion_progress: Option<f32>,
    /// Fase actual de conversión
    pub conversion_phase: Option<String>,
    /// Fecha de creación
    pub created_at: DateTime<Utc>,
    /// Fecha de última actualización
    pub updated_at: DateTime<Utc>,
    /// Metadatos adicionales
    pub metadata: serde_json::Value,
}

impl FractalModel {
    /// Crea un nuevo modelo fractal
    pub fn new(name: String, file_path: String, file_size: u64) -> Self {
        let now = Utc::now();
        Self {
            id: format!("fractal_models:{}", Uuid::new_v4()),
            name,
            architecture: ModelArchitecture {
                model_type: String::new(),
                n_layers: 0,
                embedding_dim: 0,
                vocab_size: 0,
                n_heads: 0,
                ffn_dim: 0,
            },
            root_node_id: None,
            status: FractalModelStatus::Uploading,
            file_path,
            file_size,
            conversion_progress: Some(0.0),
            conversion_phase: None,
            created_at: now,
            updated_at: now,
            metadata: serde_json::json!({}),
        }
    }

    /// Actualiza el estado del modelo
    pub fn update_status(&mut self, status: FractalModelStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Actualiza el progreso de conversión
    pub fn update_conversion_progress(&mut self, progress: f32, phase: Option<String>) {
        self.conversion_progress = Some(progress.clamp(0.0, 100.0));
        self.conversion_phase = phase;
        self.updated_at = Utc::now();
    }

    /// Establece la arquitectura del modelo
    pub fn set_architecture(&mut self, architecture: ModelArchitecture) {
        self.architecture = architecture;
        self.updated_at = Utc::now();
    }

    /// Establece el nodo raíz del grafo fractal
    pub fn set_root_node(&mut self, node_id: String) {
        self.root_node_id = Some(node_id);
        self.updated_at = Utc::now();
    }
}

/// Nodo del grafo fractal de un modelo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalModelNode {
    /// ID del nodo
    pub id: String,
    /// ID del modelo al que pertenece
    pub model_id: String,
    /// Embedding del nodo (representa una región del espacio de parámetros)
    pub embedding: Vec<f32>,
    /// Información de la capa del modelo
    pub layer_info: LayerInfo,
    /// ID del nodo padre (None si es raíz)
    pub parent_id: Option<String>,
    /// IDs de nodos hijos
    pub children_ids: Vec<String>,
    /// Nivel en el árbol fractal (0 = raíz)
    pub level: u32,
}

/// Información sobre una capa o región del modelo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Rango de capas que representa este nodo
    pub layer_range: (u32, u32),
    /// Tipo de capa (attention, ffn, embedding, etc)
    pub layer_type: String,
    /// Resumen semántico de lo que representa
    pub summary: String,
    /// Metadatos adicionales
    pub metadata: serde_json::Value,
}

impl FractalModelNode {
    /// Crea un nuevo nodo fractal
    pub fn new(
        model_id: String,
        embedding: Vec<f32>,
        layer_info: LayerInfo,
        parent_id: Option<String>,
        level: u32,
    ) -> Self {
        Self {
            id: format!("fractal_model_nodes:{}", Uuid::new_v4()),
            model_id,
            embedding,
            layer_info,
            parent_id,
            children_ids: Vec::new(),
            level,
        }
    }

    /// Añade un hijo al nodo
    pub fn add_child(&mut self, child_id: String) {
        if !self.children_ids.contains(&child_id) {
            self.children_ids.push(child_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_model_creation() {
        let model = FractalModel::new(
            "llama-2-7b".to_string(),
            "/models/llama-2-7b.gguf".to_string(),
            7_000_000_000,
        );

        assert_eq!(model.name, "llama-2-7b");
        assert_eq!(model.status, FractalModelStatus::Uploading);
        assert!(model.root_node_id.is_none());
    }

    #[test]
    fn test_model_status_update() {
        let mut model = FractalModel::new(
            "test-model".to_string(),
            "/models/test.gguf".to_string(),
            1000,
        );

        model.update_status(FractalModelStatus::Converting);
        assert_eq!(model.status, FractalModelStatus::Converting);

        model.update_status(FractalModelStatus::Ready);
        assert_eq!(model.status, FractalModelStatus::Ready);
    }

    #[test]
    fn test_fractal_node_hierarchy() {
        let mut parent = FractalModelNode::new(
            "model:123".to_string(),
            vec![0.1, 0.2, 0.3],
            LayerInfo {
                layer_range: (0, 10),
                layer_type: "attention".to_string(),
                summary: "Attention layers".to_string(),
                metadata: serde_json::json!({}),
            },
            None,
            0,
        );

        let child_id = "node:456".to_string();
        parent.add_child(child_id.clone());

        assert_eq!(parent.children_ids.len(), 1);
        assert_eq!(parent.children_ids[0], child_id);
    }
}
