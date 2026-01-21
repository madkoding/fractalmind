#![allow(dead_code)]

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tracing::{info, error};

use crate::db::connection::DatabaseConnection;
use crate::models::llm::{FractalModel, FractalModelStatus, GGUFParser, ModelArchitecture};

/// Servicio para convertir modelos GGUF a estructura fractal
pub struct ModelConversionService {
    db: Arc<DatabaseConnection>,
}

impl ModelConversionService {
    pub fn new(db: Arc<DatabaseConnection>) -> Self {
        Self { db }
    }

    /// Inicia la conversión de un modelo GGUF a estructura fractal
    pub async fn convert_model(&self, model: &mut FractalModel) -> Result<()> {
        info!("Starting conversion of model: {} ({})", model.name, model.id);

        // Actualizar estado a Converting
        model.update_status(FractalModelStatus::Converting);
        self.save_model(model).await?;

        // Parsear archivo GGUF
        let architecture = match self.parse_gguf_file(&model.file_path).await {
            Ok(arch) => arch,
            Err(e) => {
                error!("Failed to parse GGUF file: {}", e);
                model.update_status(FractalModelStatus::Failed);
                self.save_model(model).await?;
                return Err(e);
            }
        };

        model.set_architecture(architecture);
        self.save_model(model).await?;

        // TODO: Extraer embeddings de capas del modelo
        // TODO: Aplicar RAPTOR para clustering jerárquico
        // TODO: Crear grafo fractal en SurrealDB
        // TODO: Crear índice HNSW

        // Por ahora, marcar como Ready después de parsear
        info!("Model conversion completed: {}", model.name);
        model.update_status(FractalModelStatus::Ready);
        self.save_model(model).await?;

        Ok(())
    }

    /// Parsea un archivo GGUF y extrae su arquitectura
    async fn parse_gguf_file(&self, file_path: &str) -> Result<ModelArchitecture> {
        info!("Parsing GGUF file: {}", file_path);

        // Verificar que el archivo existe
        if !Path::new(file_path).exists() {
            return Err(anyhow::anyhow!("GGUF file not found: {}", file_path));
        }

        // Parsear archivo (esto es I/O bloqueante, ejecutar en thread pool)
        let path = file_path.to_string();
        let parser = tokio::task::spawn_blocking(move || {
            GGUFParser::parse_file(&path)
        })
        .await
        .context("Failed to spawn blocking task")?
        .context("Failed to parse GGUF file")?;

        // Extraer arquitectura
        let architecture = parser.extract_architecture()
            .context("Failed to extract architecture from GGUF metadata")?;

        info!(
            "Parsed GGUF architecture: {} with {} layers, dim={}",
            architecture.model_type, architecture.n_layers, architecture.embedding_dim
        );

        Ok(architecture)
    }

    /// Guarda el modelo en SurrealDB
    async fn save_model(&self, model: &FractalModel) -> Result<()> {
        let query = r#"
            UPDATE $model_id SET
                name = $name,
                architecture = $architecture,
                root_node_id = $root_node_id,
                status = $status,
                file_path = $file_path,
                file_size = $file_size,
                updated_at = time::now(),
                metadata = $metadata
        "#;

        self.db
            .query(query)
            .bind(("model_id", model.id.clone()))
            .bind(("name", model.name.clone()))
            .bind(("architecture", serde_json::to_value(&model.architecture)?))
            .bind(("root_node_id", model.root_node_id.clone()))
            .bind(("status", format!("{:?}", model.status).to_lowercase()))
            .bind(("file_path", model.file_path.clone()))
            .bind(("file_size", model.file_size))
            .bind(("metadata", model.metadata.clone()))
            .await
            .context("Failed to save model to database")?;

        Ok(())
    }

    /// Crea un nuevo modelo en la base de datos
    pub async fn create_model(&self, model: &FractalModel) -> Result<()> {
        let query = r#"
            CREATE $model_id SET
                name = $name,
                architecture = $architecture,
                root_node_id = $root_node_id,
                status = $status,
                file_path = $file_path,
                file_size = $file_size,
                created_at = time::now(),
                updated_at = time::now(),
                metadata = $metadata
        "#;

        self.db
            .query(query)
            .bind(("model_id", model.id.clone()))
            .bind(("name", model.name.clone()))
            .bind(("architecture", serde_json::to_value(&model.architecture)?))
            .bind(("root_node_id", model.root_node_id.clone()))
            .bind(("status", format!("{:?}", model.status).to_lowercase()))
            .bind(("file_path", model.file_path.clone()))
            .bind(("file_size", model.file_size))
            .bind(("metadata", model.metadata.clone()))
            .await
            .context("Failed to create model in database")?;

        Ok(())
    }

    /// Lista todos los modelos
    pub async fn list_models(&self) -> Result<Vec<FractalModel>> {
        let query = "SELECT * FROM fractal_models ORDER BY created_at DESC";
        
        let mut response = self.db
            .query(query)
            .await
            .context("Failed to list models")?;

        let models: Vec<FractalModel> = response.take(0)?;
        Ok(models)
    }

    /// Obtiene un modelo por ID
    pub async fn get_model(&self, model_id: &str) -> Result<Option<FractalModel>> {
        let query = "SELECT * FROM $model_id";
        
        let mut response = self.db
            .query(query)
            .bind(("model_id", model_id))
            .await
            .context("Failed to get model")?;

        let models: Vec<FractalModel> = response.take(0)?;
        Ok(models.into_iter().next())
    }

    /// Elimina un modelo
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        // Obtener información del modelo
        if let Some(model) = self.get_model(model_id).await? {
            // Eliminar archivo GGUF si existe
            if Path::new(&model.file_path).exists() {
                fs::remove_file(&model.file_path)
                    .await
                    .context("Failed to delete GGUF file")?;
            }
        }

        // Eliminar nodos fractales asociados
        let delete_nodes_query = "DELETE fractal_model_nodes WHERE model_id = $model_id";
        self.db
            .query(delete_nodes_query)
            .bind(("model_id", model_id))
            .await?;

        // Eliminar modelo
        let delete_model_query = "DELETE $model_id";
        self.db
            .query(delete_model_query)
            .bind(("model_id", model_id))
            .await
            .context("Failed to delete model from database")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_conversion_service_creation() {
        // Este test requeriría un mock de DB
        // Por ahora solo verificamos que compila
    }
}
