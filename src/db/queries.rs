#![allow(dead_code)]

use super::connection::DatabaseConnection;
use crate::models::{FractalNode, FractalEdge, NodeStatus};
use crate::models::llm::fractal_model::{FractalModel, FractalModelStatus, FractalModelNode};
use anyhow::{Context, Result};
use surrealdb::sql::Thing;

/// Repositorio de nodos
pub struct NodeRepository<'a> {
    db: &'a DatabaseConnection,
}

impl<'a> NodeRepository<'a> {
    pub fn new(db: &'a DatabaseConnection) -> Self {
        Self { db }
    }

    /// Crea un nuevo nodo
    pub async fn create(&self, node: &FractalNode) -> Result<Thing> {
        let result = self
            .db
            .create("nodes")
            .content(node)
            .await;
        
        match result {
            Ok(created) => {
                let nodes: Vec<FractalNode> = created;
                nodes
                    .first()
                    .and_then(|n| n.id.clone())
                    .context("Failed to get created node ID")
            }
            Err(e) => {
                tracing::error!("SurrealDB create error: {:?}", e);
                Err(anyhow::anyhow!("Failed to create node: {}", e))
            }
        }
    }

    /// Obtiene un nodo por ID
    pub async fn get_by_id(&self, id: &Thing) -> Result<Option<FractalNode>> {
        let node: Option<FractalNode> = self
            .db
            .select(id)
            .await
            .context("Failed to get node by ID")?;

        Ok(node)
    }

    /// Obtiene nodos por namespace
    pub async fn get_by_namespace(&self, namespace: &str) -> Result<Vec<FractalNode>> {
        let query = "SELECT * FROM nodes WHERE namespace = $namespace";
        let mut result = self
            .db
            .query(query)
            .bind(("namespace", namespace))
            .await
            .context("Failed to query nodes by namespace")?;

        let nodes: Vec<FractalNode> = result.take(0)?;
        Ok(nodes)
    }

    /// Obtiene nodos por estado
    pub async fn get_by_status(&self, status: NodeStatus) -> Result<Vec<FractalNode>> {
        let status_str = match status {
            NodeStatus::Complete => "complete",
            NodeStatus::Incomplete => "incomplete",
            NodeStatus::Pending => "pending",
            NodeStatus::Deprecated => "deprecated",
        };

        let query = "SELECT * FROM nodes WHERE status = $status";
        let mut result = self
            .db
            .query(query)
            .bind(("status", status_str))
            .await
            .context("Failed to query nodes by status")?;

        let nodes: Vec<FractalNode> = result.take(0)?;
        Ok(nodes)
    }

    /// Actualiza un nodo
    pub async fn update(&self, id: &Thing, node: &FractalNode) -> Result<()> {
        let _: Option<FractalNode> = self.db
            .update(id)
            .content(node)
            .await
            .context("Failed to update node")?;

        Ok(())
    }

    /// Elimina un nodo
    pub async fn delete(&self, id: &Thing) -> Result<()> {
        let _: Option<FractalNode> = self
            .db
            .delete(id)
            .await
            .context("Failed to delete node")?;

        Ok(())
    }

    /// Búsqueda vectorial por similitud (usando índice HNSW)
    pub async fn search_similar(
        &self,
        query_vector: &[f32],
        namespace: &str,
        limit: usize,
    ) -> Result<Vec<(FractalNode, f32)>> {
        // SurrealDB query para búsqueda vectorial con distancia coseno
        let query = r#"
            SELECT *, vector::similarity::cosine(embedding.vector, $query_vector) AS similarity
            FROM nodes
            WHERE namespace = $namespace
            ORDER BY similarity DESC
            LIMIT $limit
        "#;

        let mut result = self
            .db
            .query(query)
            .bind(("query_vector", query_vector))
            .bind(("namespace", namespace))
            .bind(("limit", limit))
            .await
            .context("Failed to search similar nodes")?;

        // Parse results
        #[derive(serde::Deserialize)]
        struct NodeWithSimilarity {
            #[serde(flatten)]
            node: FractalNode,
            similarity: f32,
        }

        let results: Vec<NodeWithSimilarity> = result.take(0)?;
        let nodes_with_scores = results
            .into_iter()
            .map(|nws| (nws.node, nws.similarity))
            .collect();

        Ok(nodes_with_scores)
    }

    /// Obtiene nodos "calientes" (más accedidos) para cache LRU
    pub async fn get_hot_nodes(&self, limit: usize) -> Result<Vec<FractalNode>> {
        let query = r#"
            SELECT * FROM nodes
            WHERE metadata.access_count > 10
            ORDER BY metadata.access_count DESC
            LIMIT $limit
        "#;

        let mut result = self
            .db
            .query(query)
            .bind(("limit", limit))
            .await
            .context("Failed to get hot nodes")?;

        let nodes: Vec<FractalNode> = result.take(0)?;
        Ok(nodes)
    }
}

/// Repositorio de aristas
pub struct EdgeRepository<'a> {
    db: &'a DatabaseConnection,
}

impl<'a> EdgeRepository<'a> {
    pub fn new(db: &'a DatabaseConnection) -> Self {
        Self { db }
    }

    /// Crea una nueva arista
    pub async fn create(&self, edge: &FractalEdge) -> Result<Thing> {
        let result = self
            .db
            .create("edges")
            .content(edge)
            .await;
        
        match result {
            Ok(created) => {
                let edges: Vec<FractalEdge> = created;
                edges
                    .first()
                    .and_then(|e| e.id.clone())
                    .context("Failed to get created edge ID")
            }
            Err(e) => {
                tracing::error!("SurrealDB edge create error: {:?}", e);
                tracing::error!("Edge data: from={:?}, to={:?}, type={:?}", 
                    edge.from, edge.to, edge.edge_type);
                Err(anyhow::anyhow!("Failed to create edge: {}", e))
            }
        }
    }

    /// Obtiene aristas salientes de un nodo
    pub async fn get_outgoing(&self, from: &Thing) -> Result<Vec<FractalEdge>> {
        let query = "SELECT * FROM edges WHERE from = $from";
        let mut result = self
            .db
            .query(query)
            .bind(("from", from))
            .await
            .context("Failed to get outgoing edges")?;

        let edges: Vec<FractalEdge> = result.take(0)?;
        Ok(edges)
    }

    /// Obtiene aristas entrantes a un nodo
    pub async fn get_incoming(&self, to: &Thing) -> Result<Vec<FractalEdge>> {
        let query = "SELECT * FROM edges WHERE to = $to";
        let mut result = self
            .db
            .query(query)
            .bind(("to", to))
            .await
            .context("Failed to get incoming edges")?;

        let edges: Vec<FractalEdge> = result.take(0)?;
        Ok(edges)
    }

    /// Elimina una arista
    pub async fn delete(&self, id: &Thing) -> Result<()> {
        let _: Option<FractalEdge> = self
            .db
            .delete(id)
            .await
            .context("Failed to delete edge")?;

        Ok(())
    }
}

// ============================================================================
// Fractal Model Repository
// ============================================================================

/// Repositorio de modelos fractales
pub struct FractalModelRepository<'a> {
    db: &'a DatabaseConnection,
}

impl<'a> FractalModelRepository<'a> {
    pub fn new(db: &'a DatabaseConnection) -> Self {
        Self { db }
    }

    /// Crea un nuevo modelo fractal
    pub async fn create(&self, model: &FractalModel) -> Result<String> {
        // Extract just the ID part without the table prefix
        let id_part = model.id.strip_prefix("fractal_models:").unwrap_or(&model.id);
        
        let query = r#"
            CREATE type::thing("fractal_models", $id) SET
                name = $name,
                architecture = $architecture,
                root_node_id = $root_node_id,
                status = $status,
                file_path = $file_path,
                file_size = $file_size,
                conversion_progress = $conversion_progress,
                conversion_phase = $conversion_phase,
                created_at = type::datetime($created_at),
                updated_at = type::datetime($updated_at),
                metadata = $metadata
        "#;

        let status_str = match model.status {
            FractalModelStatus::Uploading => "uploading",
            FractalModelStatus::Uploaded => "uploaded",
            FractalModelStatus::Converting => "converting",
            FractalModelStatus::Ready => "ready",
            FractalModelStatus::Failed => "failed",
        };

        let _result = self
            .db
            .query(query)
            .bind(("id", id_part))
            .bind(("name", &model.name))
            .bind(("architecture", &model.architecture))
            .bind(("root_node_id", &model.root_node_id))
            .bind(("status", status_str))
            .bind(("file_path", &model.file_path))
            .bind(("file_size", model.file_size as i64))
            .bind(("conversion_progress", model.conversion_progress))
            .bind(("conversion_phase", &model.conversion_phase))
            .bind(("created_at", model.created_at.to_rfc3339()))
            .bind(("updated_at", model.updated_at.to_rfc3339()))
            .bind(("metadata", &model.metadata))
            .await
            .context("Failed to create fractal model")?;

        // Return the full ID
        Ok(model.id.clone())
    }

    /// Obtiene un modelo por ID
    pub async fn get_by_id(&self, id: &str) -> Result<Option<FractalModel>> {
        // Extract just the ID part without the table prefix
        let id_part = id.strip_prefix("fractal_models:").unwrap_or(id);
        
        let query = "SELECT * FROM type::thing(\"fractal_models\", $id)";
        let mut result = self
            .db
            .query(query)
            .bind(("id", id_part))
            .await
            .context("Failed to get fractal model by ID")?;

        let models: Vec<FractalModel> = result.take(0)?;
        Ok(models.into_iter().next())
    }

    /// Lista todos los modelos
    pub async fn list_all(&self) -> Result<Vec<FractalModel>> {
        let query = "SELECT * FROM fractal_models ORDER BY created_at DESC";
        let mut result = self
            .db
            .query(query)
            .await
            .context("Failed to list fractal models")?;

        let models: Vec<FractalModel> = result.take(0)?;
        Ok(models)
    }

    /// Lista modelos por estado
    pub async fn list_by_status(&self, status: FractalModelStatus) -> Result<Vec<FractalModel>> {
        let status_str = match status {
            FractalModelStatus::Uploading => "uploading",
            FractalModelStatus::Uploaded => "uploaded",
            FractalModelStatus::Converting => "converting",
            FractalModelStatus::Ready => "ready",
            FractalModelStatus::Failed => "failed",
        };

        let query = "SELECT * FROM fractal_models WHERE status = $status ORDER BY created_at DESC";
        let mut result = self
            .db
            .query(query)
            .bind(("status", status_str))
            .await
            .context("Failed to list fractal models by status")?;

        let models: Vec<FractalModel> = result.take(0)?;
        Ok(models)
    }

    /// Actualiza el estado de un modelo
    pub async fn update_status(&self, id: &str, status: FractalModelStatus) -> Result<()> {
        let status_str = match status {
            FractalModelStatus::Uploading => "uploading",
            FractalModelStatus::Uploaded => "uploaded",
            FractalModelStatus::Converting => "converting",
            FractalModelStatus::Ready => "ready",
            FractalModelStatus::Failed => "failed",
        };

        // Extract just the ID part without the table prefix
        let id_part = id.strip_prefix("fractal_models:").unwrap_or(id);

        let query = r#"
            UPDATE type::thing("fractal_models", $id) SET 
                status = $status,
                updated_at = time::now()
        "#;

        self.db
            .query(query)
            .bind(("id", id_part))
            .bind(("status", status_str))
            .await
            .context("Failed to update fractal model status")?;

        Ok(())
    }

    /// Actualiza el progreso de conversión
    pub async fn update_conversion_progress(
        &self,
        id: &str,
        progress: f32,
        phase: Option<&str>,
    ) -> Result<()> {
        // Extract just the ID part without the table prefix
        let id_part = id.strip_prefix("fractal_models:").unwrap_or(id);

        let query = r#"
            UPDATE type::thing("fractal_models", $id) SET 
                conversion_progress = $progress,
                conversion_phase = $phase,
                updated_at = time::now()
        "#;

        self.db
            .query(query)
            .bind(("id", id_part))
            .bind(("progress", progress))
            .bind(("phase", phase))
            .await
            .context("Failed to update conversion progress")?;

        Ok(())
    }

    /// Actualiza la arquitectura del modelo
    pub async fn update_architecture(
        &self,
        id: &str,
        architecture: &crate::models::llm::fractal_model::ModelArchitecture,
    ) -> Result<()> {
        // Extract just the ID part without the table prefix
        let id_part = id.strip_prefix("fractal_models:").unwrap_or(id);

        let query = r#"
            UPDATE type::thing("fractal_models", $id) SET 
                architecture = $architecture,
                updated_at = time::now()
        "#;

        self.db
            .query(query)
            .bind(("id", id_part))
            .bind(("architecture", architecture))
            .await
            .context("Failed to update model architecture")?;

        Ok(())
    }

    /// Establece el nodo raíz del grafo fractal
    pub async fn set_root_node(&self, id: &str, root_node_id: &str) -> Result<()> {
        // Extract just the ID part without the table prefix
        let id_part = id.strip_prefix("fractal_models:").unwrap_or(id);

        let query = r#"
            UPDATE type::thing("fractal_models", $id) SET 
                root_node_id = $root_node_id,
                updated_at = time::now()
        "#;

        self.db
            .query(query)
            .bind(("id", id_part))
            .bind(("root_node_id", root_node_id))
            .await
            .context("Failed to set root node")?;

        Ok(())
    }

    /// Elimina un modelo
    pub async fn delete(&self, id: &str) -> Result<Option<FractalModel>> {
        // First get the model to return it
        let model = self.get_by_id(id).await?;
        
        // Extract just the ID part without the table prefix
        let id_part = id.strip_prefix("fractal_models:").unwrap_or(id);
        
        // Delete from database using type::thing
        let query = "DELETE type::thing(\"fractal_models\", $id)";
        self.db
            .query(query)
            .bind(("id", id_part))
            .await
            .context("Failed to delete fractal model")?;

        // Also delete associated nodes
        let delete_nodes_query = "DELETE FROM fractal_model_nodes WHERE model_id = $model_id";
        self.db
            .query(delete_nodes_query)
            .bind(("model_id", id))
            .await
            .context("Failed to delete fractal model nodes")?;

        Ok(model)
    }
}

// ============================================================================
// Fractal Model Node Repository
// ============================================================================

/// Repositorio de nodos de modelos fractales
pub struct FractalModelNodeRepository<'a> {
    db: &'a DatabaseConnection,
}

impl<'a> FractalModelNodeRepository<'a> {
    pub fn new(db: &'a DatabaseConnection) -> Self {
        Self { db }
    }

    /// Crea un nuevo nodo de modelo fractal
    pub async fn create(&self, node: &FractalModelNode) -> Result<String> {
        let query = r#"
            CREATE fractal_model_nodes SET
                model_id = $model_id,
                embedding = $embedding,
                layer_info = $layer_info,
                parent_id = $parent_id,
                children_ids = $children_ids,
                level = $level
        "#;

        let mut result = self
            .db
            .query(query)
            .bind(("model_id", &node.model_id))
            .bind(("embedding", &node.embedding))
            .bind(("layer_info", &node.layer_info))
            .bind(("parent_id", &node.parent_id))
            .bind(("children_ids", &node.children_ids))
            .bind(("level", node.level as i64))
            .await
            .context("Failed to create fractal model node")?;

        let created: Vec<serde_json::Value> = result.take(0)?;
        let id = created
            .first()
            .and_then(|v| v.get("id"))
            .and_then(|id| id.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| node.id.clone());

        Ok(id)
    }

    /// Crea múltiples nodos en batch
    pub async fn create_batch(&self, nodes: &[FractalModelNode]) -> Result<Vec<String>> {
        let mut ids = Vec::with_capacity(nodes.len());
        
        for node in nodes {
            let id = self.create(node).await?;
            ids.push(id);
        }
        
        Ok(ids)
    }

    /// Obtiene nodos por modelo
    pub async fn get_by_model(&self, model_id: &str) -> Result<Vec<FractalModelNode>> {
        let query = "SELECT * FROM fractal_model_nodes WHERE model_id = $model_id ORDER BY level";
        let mut result = self
            .db
            .query(query)
            .bind(("model_id", model_id))
            .await
            .context("Failed to get nodes by model")?;

        let nodes: Vec<FractalModelNode> = result.take(0)?;
        Ok(nodes)
    }

    /// Obtiene el nodo raíz de un modelo
    pub async fn get_root(&self, model_id: &str) -> Result<Option<FractalModelNode>> {
        let query = r#"
            SELECT * FROM fractal_model_nodes 
            WHERE model_id = $model_id AND parent_id IS NONE
            LIMIT 1
        "#;

        let mut result = self
            .db
            .query(query)
            .bind(("model_id", model_id))
            .await
            .context("Failed to get root node")?;

        let nodes: Vec<FractalModelNode> = result.take(0)?;
        Ok(nodes.into_iter().next())
    }

    /// Obtiene hijos de un nodo
    pub async fn get_children(&self, node_id: &str) -> Result<Vec<FractalModelNode>> {
        let query = "SELECT * FROM fractal_model_nodes WHERE parent_id = $node_id";
        let mut result = self
            .db
            .query(query)
            .bind(("node_id", node_id))
            .await
            .context("Failed to get children")?;

        let nodes: Vec<FractalModelNode> = result.take(0)?;
        Ok(nodes)
    }

    /// Búsqueda por similitud de embedding
    pub async fn search_similar(
        &self,
        embedding: &[f32],
        model_id: &str,
        limit: usize,
    ) -> Result<Vec<(FractalModelNode, f32)>> {
        let query = r#"
            SELECT *, vector::similarity::cosine(embedding, $embedding) AS similarity
            FROM fractal_model_nodes
            WHERE model_id = $model_id
            ORDER BY similarity DESC
            LIMIT $limit
        "#;

        let mut result = self
            .db
            .query(query)
            .bind(("embedding", embedding))
            .bind(("model_id", model_id))
            .bind(("limit", limit))
            .await
            .context("Failed to search similar nodes")?;

        #[derive(serde::Deserialize)]
        struct NodeWithSimilarity {
            #[serde(flatten)]
            node: FractalModelNode,
            similarity: f32,
        }

        let results: Vec<NodeWithSimilarity> = result.take(0)?;
        let nodes_with_scores = results
            .into_iter()
            .map(|nws| (nws.node, nws.similarity))
            .collect();

        Ok(nodes_with_scores)
    }

    /// Cuenta nodos por nivel
    pub async fn count_by_level(&self, model_id: &str) -> Result<Vec<(u32, usize)>> {
        let query = r#"
            SELECT level, count() AS count
            FROM fractal_model_nodes
            WHERE model_id = $model_id
            GROUP BY level
            ORDER BY level
        "#;

        let mut result = self
            .db
            .query(query)
            .bind(("model_id", model_id))
            .await
            .context("Failed to count by level")?;

        #[derive(serde::Deserialize)]
        struct LevelCount {
            level: u32,
            count: usize,
        }

        let counts: Vec<LevelCount> = result.take(0)?;
        Ok(counts.into_iter().map(|lc| (lc.level, lc.count)).collect())
    }

    /// Actualiza el parent_id de un nodo
    pub async fn update_parent(&self, node_id: &str, parent_id: &str) -> Result<()> {
        let query = r#"
            UPDATE fractal_model_nodes SET 
                parent_id = $parent_id
            WHERE id = $node_id OR string::split(type::string(id), ':')[1] = $node_id
        "#;

        self.db
            .query(query)
            .bind(("node_id", node_id))
            .bind(("parent_id", parent_id))
            .await
            .context("Failed to update parent")?;

        Ok(())
    }

    /// Elimina todos los nodos de un modelo
    pub async fn delete_by_model(&self, model_id: &str) -> Result<usize> {
        let query = "DELETE FROM fractal_model_nodes WHERE model_id = $model_id RETURN BEFORE";
        let mut result = self
            .db
            .query(query)
            .bind(("model_id", model_id))
            .await
            .context("Failed to delete nodes by model")?;

        let deleted: Vec<serde_json::Value> = result.take(0)?;
        Ok(deleted.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{NodeMetadata, EmbeddingVector, EmbeddingModel};

    fn create_test_node() -> FractalNode {
        let embedding = EmbeddingVector::new(
            vec![0.1; 768],
            EmbeddingModel::NomicEmbedTextV15,
        );

        FractalNode::new_leaf(
            "Test content".to_string(),
            embedding,
            "global_knowledge".to_string(),
            None,
            NodeMetadata::default(),
        )
    }

    #[tokio::test]
    #[ignore]
    async fn test_create_node() {
        dotenv::dotenv().ok();
        let config = crate::db::connection::DbConfig::from_env().unwrap();
        let db = crate::db::connection::connect_db(&config).await.unwrap();

        let repo = NodeRepository::new(&db);
        let node = create_test_node();

        let result = repo.create(&node).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_get_by_status() {
        dotenv::dotenv().ok();
        let config = crate::db::connection::DbConfig::from_env().unwrap();
        let db = crate::db::connection::connect_db(&config).await.unwrap();

        let repo = NodeRepository::new(&db);
        let result = repo.get_by_status(NodeStatus::Incomplete).await;
        assert!(result.is_ok());
    }
}
