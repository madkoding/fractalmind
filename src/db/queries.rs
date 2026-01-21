#![allow(dead_code)]

use super::connection::DatabaseConnection;
use crate::models::{FractalNode, FractalEdge, NodeStatus};
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
        let created: Vec<FractalNode> = self
            .db
            .create("nodes")
            .content(node)
            .await
            .context("Failed to create node")?;

        created
            .first()
            .and_then(|n| n.id.clone())
            .context("Failed to get created node ID")
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
        let created: Vec<FractalEdge> = self
            .db
            .create("edges")
            .content(edge)
            .await
            .context("Failed to create edge")?;

        created
            .first()
            .and_then(|e| e.id.clone())
            .context("Failed to get created edge ID")
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
