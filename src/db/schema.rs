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
    
    // Definir tablas para modelos fractales
    define_fractal_models_tables(db).await?;

    info!("Database schema initialized successfully");

    Ok(())
}

/// Define la tabla de nodos fractales
async fn define_nodes_table(db: &DatabaseConnection) -> Result<()> {
    // Use SCHEMALESS to allow flexible object fields for embedding and metadata
    let query = r#"
        DEFINE TABLE nodes SCHEMALESS;

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
    // Use SCHEMALESS to allow flexible serialization from Rust types
    // SurrealDB SCHEMAFULL requires native datetime types which are complex
    // to serialize correctly from chrono::DateTime
    let query = r#"
        DEFINE TABLE edges SCHEMALESS;

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
    // Usar solo el índice principal para Nomic (768 dims) por ahora
    // Los demás modelos pueden agregarse cuando se usen
    let query = r#"
        DEFINE INDEX idx_embedding_vector ON TABLE nodes
            FIELDS embedding.vector
            MTREE DIMENSION 768
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

/// Define tablas para modelos fractales
async fn define_fractal_models_tables(db: &DatabaseConnection) -> Result<()> {
    let query = r#"
        DEFINE TABLE fractal_models SCHEMALESS;

        DEFINE INDEX idx_fractal_models_name ON TABLE fractal_models COLUMNS name;
        DEFINE INDEX idx_fractal_models_status ON TABLE fractal_models COLUMNS status;

        DEFINE TABLE fractal_model_nodes SCHEMALESS;

        DEFINE INDEX idx_fractal_nodes_model ON TABLE fractal_model_nodes COLUMNS model_id;
        DEFINE INDEX idx_fractal_nodes_level ON TABLE fractal_model_nodes COLUMNS level;
        
        DEFINE TABLE upload_sessions SCHEMALESS;

        DEFINE INDEX idx_upload_sessions_id ON TABLE upload_sessions COLUMNS upload_id UNIQUE;
        DEFINE INDEX idx_upload_sessions_status ON TABLE upload_sessions COLUMNS status;
    "#;

    db.query(query)
        .await
        .context("Failed to define fractal models tables")?;

    info!("Fractal models tables defined");
    Ok(())
}

/// Inicializa el esquema de modelos fractales
#[allow(dead_code)]
pub async fn initialize_fractal_models_schema(db: &DatabaseConnection) -> Result<()> {
    info!("Initializing fractal models schema...");
    define_fractal_models_tables(db).await?;
    info!("Fractal models schema initialized successfully");
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
