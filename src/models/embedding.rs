#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Modelos de embedding soportados
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingModel {
    /// Nomic Embed Text v1.5 (768 dimensiones)
    NomicEmbedTextV15,
    /// BAAI BGE Small (384 dimensiones)
    BaaiGgeSmall,
    /// Sentence Transformers all-MiniLM-L6-v2 (384 dimensiones)
    AllMiniLmL6V2,
    /// CLIP para embeddings multimodales (512 dimensiones)
    ClipVitB32,
    /// Modelo personalizado
    Custom(String),
}

impl EmbeddingModel {
    /// Retorna la dimensión esperada del vector
    pub fn dimension(&self) -> usize {
        match self {
            EmbeddingModel::NomicEmbedTextV15 => 768,
            EmbeddingModel::BaaiGgeSmall => 384,
            EmbeddingModel::AllMiniLmL6V2 => 384,
            EmbeddingModel::ClipVitB32 => 512,
            EmbeddingModel::Custom(_) => 0, // Debe especificarse manualmente
        }
    }

    /// Retorna el nombre del modelo para fastembed
    pub fn model_name(&self) -> &str {
        match self {
            EmbeddingModel::NomicEmbedTextV15 => "nomic-ai/nomic-embed-text-v1.5",
            EmbeddingModel::BaaiGgeSmall => "BAAI/bge-small-en-v1.5",
            EmbeddingModel::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingModel::ClipVitB32 => "openai/clip-vit-base-patch32",
            EmbeddingModel::Custom(name) => name,
        }
    }

    /// Verifica si el modelo soporta imágenes
    pub fn supports_images(&self) -> bool {
        matches!(self, EmbeddingModel::ClipVitB32 | EmbeddingModel::Custom(_))
    }
}

/// Vector de embedding con metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVector {
    /// Vector de embeddings (valores entre -1.0 y 1.0, normalizados)
    pub vector: Vec<f32>,

    /// Modelo usado para generar el embedding
    pub model: EmbeddingModel,

    /// Dimensión del vector
    pub dimension: usize,
}

impl EmbeddingVector {
    /// Crea un nuevo vector de embedding
    pub fn new(vector: Vec<f32>, model: EmbeddingModel) -> Self {
        let dimension = vector.len();
        Self {
            vector,
            model,
            dimension,
        }
    }

    /// Calcula similitud coseno con otro vector
    pub fn cosine_similarity(&self, other: &EmbeddingVector) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Calcula distancia euclidiana con otro vector
    pub fn euclidean_distance(&self, other: &EmbeddingVector) -> f32 {
        if self.dimension != other.dimension {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normaliza el vector (magnitud = 1)
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut self.vector {
                *value /= norm;
            }
        }
    }

    /// Verifica si el vector está normalizado
    pub fn is_normalized(&self) -> bool {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        (norm - 1.0).abs() < 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_model_dimensions() {
        assert_eq!(EmbeddingModel::NomicEmbedTextV15.dimension(), 768);
        assert_eq!(EmbeddingModel::BaaiGgeSmall.dimension(), 384);
        assert_eq!(EmbeddingModel::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(EmbeddingModel::ClipVitB32.dimension(), 512);
    }

    #[test]
    fn test_cosine_similarity() {
        let vec1 = EmbeddingVector::new(
            vec![1.0, 0.0, 0.0],
            EmbeddingModel::NomicEmbedTextV15,
        );
        let vec2 = EmbeddingVector::new(
            vec![1.0, 0.0, 0.0],
            EmbeddingModel::NomicEmbedTextV15,
        );
        let vec3 = EmbeddingVector::new(
            vec![0.0, 1.0, 0.0],
            EmbeddingModel::NomicEmbedTextV15,
        );

        assert!((vec1.cosine_similarity(&vec2) - 1.0).abs() < 1e-6);
        assert!((vec1.cosine_similarity(&vec3) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let vec1 = EmbeddingVector::new(
            vec![0.0, 0.0, 0.0],
            EmbeddingModel::NomicEmbedTextV15,
        );
        let vec2 = EmbeddingVector::new(
            vec![1.0, 1.0, 1.0],
            EmbeddingModel::NomicEmbedTextV15,
        );

        let distance = vec1.euclidean_distance(&vec2);
        assert!((distance - 3.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut vec = EmbeddingVector::new(
            vec![3.0, 4.0, 0.0],
            EmbeddingModel::NomicEmbedTextV15,
        );

        assert!(!vec.is_normalized());
        vec.normalize();
        assert!(vec.is_normalized());

        // Verificar valores normalizados (3-4-5 triangle)
        assert!((vec.vector[0] - 0.6).abs() < 1e-6);
        assert!((vec.vector[1] - 0.8).abs() < 1e-6);
        assert!((vec.vector[2] - 0.0).abs() < 1e-6);
    }
}
