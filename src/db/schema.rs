use super::connection::DatabaseConnection;
use anyhow::{Context, Result};
use tracing::info;

/// Inicializa el esquema de la base de datos
pub async fn initialize_schema(db: &DatabaseConnection) -> Result<()> {
    info!("Initializing database schema...");

    // Definir tabla de nodos fractales
    define_nodes_table(db).await?;

    // Definir tabla de aristas
    define_edges_table(db).await?;

    // Definir índices HNSW para búsqueda vectorial
    define_vector_indexes(db).await?;

    // Definir namespaces y scopes
    define_namespaces(db).await?;

    info!("Database schema initialized successfully");

    Ok(())
}

/// Define la tabla de nodos fractales
async fn define_nodes_table(db: &DatabaseConnection) -> Result<()> {
    // SurrealDB 1.5.x no soporta IF NOT EXISTS, usar OVERWRITE
    let query = r#"
        DEFINE TABLE nodes SCHEMAFULL;

        DEFINE FIELD uuid ON TABLE nodes TYPE string;
        DEFINE FIELD node_type ON TABLE nodes TYPE string
            ASSERT $value IN ["leaf", "parent", "root"];
        DEFINE FIELD status ON TABLE nodes TYPE string
            ASSERT $value IN ["complete", "incomplete", "pending", "deprecated"];
        DEFINE FIELD content ON TABLE nodes TYPE string;
        DEFINE FIELD summary ON TABLE nodes TYPE option<string>;
        DEFINE FIELD embedding ON TABLE nodes TYPE object;
        DEFINE FIELD depth_level ON TABLE nodes TYPE int;
        DEFINE FIELD confidence ON TABLE nodes TYPE float
            ASSERT $value >= 0.0 AND $value <= 1.0;
        DEFINE FIELD namespace ON TABLE nodes TYPE string;
        DEFINE FIELD scope ON TABLE nodes TYPE option<string>;
        DEFINE FIELD metadata ON TABLE nodes TYPE object;
        DEFINE FIELD created_at ON TABLE nodes TYPE datetime;
        DEFINE FIELD updated_at ON TABLE nodes TYPE datetime;
        DEFINE FIELD last_accessed_at ON TABLE nodes TYPE option<datetime>;

        DEFINE INDEX idx_uuid ON TABLE nodes COLUMNS uuid UNIQUE;
        DEFINE INDEX idx_namespace ON TABLE nodes COLUMNS namespace;
        DEFINE INDEX idx_status ON TABLE nodes COLUMNS status;
        DEFINE INDEX idx_node_type ON TABLE nodes COLUMNS node_type;
        DEFINE INDEX idx_depth_level ON TABLE nodes COLUMNS depth_level;
    "#;

    db.query(query)
        .await
        .context("Failed to define nodes table")?;

    info!("Nodes table defined");
    Ok(())
}

/// Define la tabla de aristas
async fn define_edges_table(db: &DatabaseConnection) -> Result<()> {
    let query = r#"
        DEFINE TABLE edges SCHEMAFULL;

        DEFINE FIELD from ON TABLE edges TYPE record(nodes);
        DEFINE FIELD to ON TABLE edges TYPE record(nodes);
        DEFINE FIELD edge_type ON TABLE edges TYPE string
            ASSERT $value IN ["parentchild", "semantic", "temporal", "causal", "crossnamespace", "similar"];
        DEFINE FIELD weight ON TABLE edges TYPE float;
        DEFINE FIELD similarity ON TABLE edges TYPE float
            ASSERT $value >= 0.0 AND $value <= 1.0;
        DEFINE FIELD confidence ON TABLE edges TYPE float
            ASSERT $value >= 0.0 AND $value <= 1.0;
        DEFINE FIELD created_at ON TABLE edges TYPE datetime;
        DEFINE FIELD updated_at ON TABLE edges TYPE datetime;

        DEFINE INDEX idx_from ON TABLE edges COLUMNS from;
        DEFINE INDEX idx_to ON TABLE edges COLUMNS to;
        DEFINE INDEX idx_edge_type ON TABLE edges COLUMNS edge_type;
    "#;

    db.query(query)
        .await
        .context("Failed to define edges table")?;

    info!("Edges table defined");
    Ok(())
}

/// Define índices HNSW para búsqueda vectorial eficiente
async fn define_vector_indexes(db: &DatabaseConnection) -> Result<()> {
    // SurrealDB 1.5+ soporta índices vectoriales MTREE
    let query = r#"
        DEFINE INDEX idx_embedding_nomic ON TABLE nodes
            FIELDS embedding.vector
            MTREE DIMENSION 768
            DIST COSINE
            TYPE F32;

        DEFINE INDEX idx_embedding_small ON TABLE nodes
            FIELDS embedding.vector
            MTREE DIMENSION 384
            DIST COSINE
            TYPE F32;

        DEFINE INDEX idx_embedding_clip ON TABLE nodes
            FIELDS embedding.vector
            MTREE DIMENSION 512
            DIST COSINE
            TYPE F32;
    "#;

    db.query(query)
        .await
        .context("Failed to define vector indexes")?;

    info!("Vector indexes (MTREE) defined");
    Ok(())
}

/// Define namespaces y scopes para multi-usuario
async fn define_namespaces(db: &DatabaseConnection) -> Result<()> {
    let query = r#"
        DEFINE TABLE namespaces SCHEMAFULL;
        DEFINE FIELD name ON TABLE namespaces TYPE string;
        DEFINE FIELD namespace_type ON TABLE namespaces TYPE string;
        DEFINE FIELD description ON TABLE namespaces TYPE string;
        DEFINE FIELD created_at ON TABLE namespaces TYPE datetime;

        DEFINE INDEX idx_namespace_name ON TABLE namespaces COLUMNS name UNIQUE;

        DEFINE TABLE scopes SCHEMAFULL;
        DEFINE FIELD user_id ON TABLE scopes TYPE string;
        DEFINE FIELD namespace ON TABLE scopes TYPE string;
        DEFINE FIELD permissions ON TABLE scopes TYPE object;
        DEFINE FIELD created_at ON TABLE scopes TYPE datetime;
        DEFINE FIELD expires_at ON TABLE scopes TYPE option<datetime>;

        DEFINE INDEX idx_scope_user ON TABLE scopes COLUMNS user_id;
    "#;

    db.query(query)
        .await
        .context("Failed to define namespaces")?;

    info!("Namespaces and scopes defined");
    Ok(())
}

/// Inserta el namespace global por defecto
pub async fn seed_global_namespace(db: &DatabaseConnection) -> Result<()> {
    let query = r#"
        CREATE namespaces:global SET
            name = "global_knowledge",
            namespace_type = "global",
            description = "Shared global knowledge base",
            created_at = time::now();
    "#;

    db.query(query)
        .await
        .context("Failed to seed global namespace")?;

    info!("Global namespace seeded");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Estos tests requieren una instancia de SurrealDB corriendo
    // En un entorno de CI/CD, usar un contenedor temporal

    #[tokio::test]
    #[ignore] // Ignorar por defecto, ejecutar con: cargo test -- --ignored
    async fn test_initialize_schema() {
        dotenv::dotenv().ok();
        let config = crate::db::connection::DbConfig::from_env().unwrap();
        let db = crate::db::connection::connect_db(&config).await.unwrap();

        let result = initialize_schema(&db).await;
        assert!(result.is_ok());
    }
}
