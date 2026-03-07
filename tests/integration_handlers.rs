//! Integration tests for API handlers
//!
//! These tests verify the complete flow of the implemented handlers:
//! - remember: Create episodic memory nodes
//! - memory_update: Update existing nodes
//! - stats: Get system statistics

use fractalmind::api::handlers::*;
use fractalmind::api::types::*;
use fractalmind::db::connection::{connect_db, DbConfig, DatabaseConnection};
use fractalmind::db::queries::{NodeRepository, EdgeRepository};
use fractalmind::models::{FractalNode, EmbeddingVector, NodeMetadata, EmbeddingModel};
use fractalmind::models::llm::ModelBrain;
use fractalmind::cache::{NodeCache, EmbeddingCache};
use fractalmind::api::progress::ProgressTracker;
use fractalmind::services::UploadSessionManager;

use axum::Json;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Create a test database connection
async fn setup_test_db() -> DatabaseConnection {
    let config = DbConfig {
        url: "ws://localhost:8000".to_string(),
        username: "root".to_string(),
        password: "root".to_string(),
        namespace: "test".to_string(),
        database: "fractalmind_test".to_string(),
    };
    
    connect_db(&config).await.expect("Failed to connect to test database")
}

/// Create test app state
async fn setup_test_state(db: DatabaseConnection) -> SharedState {
    let brain = ModelBrain::with_ollama_only(
        "http://localhost:11434".to_string(),
        "nomic-embed-text".to_string(),
    ).expect("Failed to create ModelBrain");
    
    let node_cache = NodeCache::with_capacity(100);
    let embedding_cache = EmbeddingCache::with_capacity(100);
    let progress_tracker = ProgressTracker::new();
    let upload_manager = Arc::new(UploadSessionManager::new("./uploads".to_string()).await);
    
    let state = AppState {
        db,
        brain,
        node_cache,
        embedding_cache,
        progress_tracker,
        upload_manager,
    };
    
    Arc::new(RwLock::new(state))
}

/// Clean up test data
async fn cleanup_test_data(db: &DatabaseConnection, node_ids: &[String]) {
    let node_repo = NodeRepository::new(db);
    
    for node_id in node_ids {
        if let Ok(thing) = parse_thing_from_string(node_id) {
            let _ = node_repo.delete(&thing).await;
        }
    }
}

// ============================================================================
// Remember Handler Tests
// ============================================================================

#[tokio::test]
async fn test_remember_handler_creates_node() {
    // Skip if no database available
    if std::env::var("SKIP_INTEGRATION_TESTS").is_ok() {
        return;
    }

    let db = setup_test_db().await;
    let state = setup_test_state(db.clone()).await;
    
    let request = RememberRequest {
        content: "Test episodic memory".to_string(),
        user_id: Some("test_user".to_string()),
        language: Some("en".to_string()),
        related_to: None,
        context: None,
    };
    
    let result = remember(state, Json(request)).await;
    
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.0.success);
    assert!(response.0.node_id.is_some());
    assert!(response.0.message.contains("Memory stored successfully"));
    
    // Cleanup
    if let Some(node_id) = response.0.node_id {
        cleanup_test_data(&db, &[node_id]).await;
    }
}

#[tokio::test]
async fn test_remember_handler_empty_content() {
    let db = setup_test_db().await;
    let state = setup_test_state(db).await;
    
    let request = RememberRequest {
        content: "".to_string(),
        user_id: None,
        language: None,
        related_to: None,
        context: None,
    };
    
    let result = remember(state, Json(request)).await;
    
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ApiError::ValidationError(_)));
}

#[tokio::test]
async fn test_remember_handler_with_namespace() {
    let db = setup_test_db().await;
    let state = setup_test_state(db.clone()).await;
    
    let request = RememberRequest {
        content: "Memory with custom namespace".to_string(),
        user_id: None,
        language: None,
        related_to: None,
        context: Some("test".to_string()),
    };
    
    let result = remember(state, Json(request)).await;
    
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.0.success);
    
    // Verify namespace is episodic when no user_id
    assert!(response.0.message.contains("episodic"));
    
    if let Some(node_id) = response.0.node_id {
        cleanup_test_data(&db, &[node_id]).await;
    }
}

// ============================================================================
// Memory Update Handler Tests
// ============================================================================

#[tokio::test]
async fn test_memory_update_handler_empty_id() {
    let db = setup_test_db().await;
    let state = setup_test_state(db).await;
    
    let request = MemoryUpdateRequest {
        node_id: "".to_string(),
        content: Some("new content".to_string()),
        status: None,
        tags: None,
        deprecated: None,
        regenerate_embedding: None,
        source: None,
        metadata: None,
    };
    
    let result = memory_update(state, Json(request)).await;
    
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ApiError::ValidationError(_)));
}

#[tokio::test]
async fn test_memory_update_handler_not_found() {
    let db = setup_test_db().await;
    let state = setup_test_state(db).await;
    
    let request = MemoryUpdateRequest {
        node_id: "nodes:nonexistent".to_string(),
        content: Some("new content".to_string()),
        status: None,
        tags: None,
        deprecated: None,
        regenerate_embedding: None,
        source: None,
        metadata: None,
    };
    
    let result = memory_update(state, Json(request)).await;
    
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ApiError::NotFound(_)));
}

#[tokio::test]
async fn test_memory_update_handler_full_update() {
    // Skip if no database available
    if std::env::var("SKIP_INTEGRATION_TESTS").is_ok() {
        return;
    }

    let db = setup_test_db().await;
    let state = setup_test_state(db.clone()).await;
    
    // First create a node to update
    let node_repo = NodeRepository::new(&db);
    let initial_node = FractalNode::new_leaf(
        "Initial content".to_string(),
        EmbeddingVector::new(vec![0.0f32; 768], EmbeddingModel::NomicEmbedTextV15),
        "test_namespace".to_string(),
        None,
        NodeMetadata::default(),
    );
    
    let created_id = node_repo.create(&initial_node).await.unwrap();
    let created_id_str = created_id.to_string();
    
    // Now update the node
    let request = MemoryUpdateRequest {
        node_id: created_id_str.clone(),
        content: Some("Updated content".to_string()),
        status: Some("complete".to_string()),
        tags: Some(vec!["updated".to_string(), "test".to_string()]),
        deprecated: Some(false),
        regenerate_embedding: Some(false), // Skip embedding regeneration in test
        source: Some("test_source".to_string()),
        metadata: Some(MetadataUpdate {
            language: Some("es".to_string()),
            access_count: Some(10),
        }),
    };
    
    let result = memory_update(state, Json(request)).await;
    
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.0.success);
    assert_eq!(response.0.node_id, created_id_str);
    assert!(!response.0.updated_fields.is_empty());
    
    // Verify the update was applied
    let updated_node = node_repo.get_by_id(&created_id).await.unwrap().unwrap();
    assert_eq!(updated_node.content, "Updated content");
    assert_eq!(updated_node.status, "complete");
    
    // Cleanup
    cleanup_test_data(&db, &[created_id_str]).await;
}

// ============================================================================
// Stats Handler Tests
// ============================================================================

#[tokio::test]
async fn test_stats_handler_returns_valid_response() {
    let db = setup_test_db().await;
    let state = setup_test_state(db).await;
    
    let response = stats(state).await;
    
    assert!(response.total_nodes >= 0);
    assert!(response.total_edges >= 0);
    assert!(response.cache_metrics.capacity > 0);
    assert!(!response.llm_metrics.embedding_model.is_empty());
}

#[tokio::test]
async fn test_stats_handler_with_data() {
    // Skip if no database available
    if std::env::var("SKIP_INTEGRATION_TESTS").is_ok() {
        return;
    }

    let db = setup_test_db().await;
    
    // Insert some test data
    let node_repo = NodeRepository::new(&db);
    let mut created_ids = Vec::new();
    
    for i in 0..3 {
        let node = FractalNode::new_leaf(
            format!("Test content {}", i),
            EmbeddingVector::new(vec![0.0f32; 768], EmbeddingModel::NomicEmbedTextV15),
            "test_stats_namespace".to_string(),
            None,
            NodeMetadata::default(),
        );
        
        let id = node_repo.create(&node).await.unwrap();
        created_ids.push(id.to_string());
    }
    
    let state = setup_test_state(db.clone()).await;
    let response = stats(state).await;
    
    // Verify stats include our test data
    assert!(response.total_nodes >= 3);
    
    // Check if our namespace is in the list
    let test_namespace = response.namespaces
        .iter()
        .find(|ns| ns.name == "test_stats_namespace");
    assert!(test_namespace.is_some());
    assert!(test_namespace.unwrap().node_count >= 3);
    
    // Cleanup
    cleanup_test_data(&db, &created_ids).await;
}

// ============================================================================
// Parse Thing Helper Tests
// ============================================================================

#[test]
fn test_parse_thing_from_string_variations() {
    // Standard format
    assert!(parse_thing_from_string("nodes:123").is_some());
    
    // Plain ID (should default to nodes table)
    assert!(parse_thing_from_string("456").is_some());
    
    // Empty string
    assert!(parse_thing_from_string("").is_some());
    
    // UUID format
    let uuid = Uuid::new_v4().to_string();
    assert!(parse_thing_from_string(&uuid).is_some());
}
