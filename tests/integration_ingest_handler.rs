use axum::{body::{self, Body}, http::{Request, header}};
use tower::util::ServiceExt;
use fractalmind::api::routes::create_router;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use fractalmind::api::handlers::AppState;
use fractalmind::models::llm::BrainConfig;
use fractalmind::models::llm::ModelBrain;
use fractalmind::cache::EmbeddingCache;
use fractalmind::cache::NodeCache;
use fractalmind::db::connection::DbConfig;

#[tokio::test]
async fn test_ingest_file_handler_multipart() {
    // Enable test helpers
    std::env::set_var("TEST_DISABLE_EMBEDDING", "true");
    std::env::set_var("TEST_SKIP_DB_WRITES", "true");
    std::env::set_var("EMBEDDING_DIMENSION", "8");

    // Build a minimal AppState. DB connection is not used because we set TEST_SKIP_DB_WRITES
    // Create a ModelBrain without health checks (providers created but not used because embedding is disabled)
    let brain = ModelBrain::new_without_health_check(BrainConfig::default_local()).expect("Failed to create brain");

    // Create dummy DB config placeholder (not connected)
    let db_cfg = DbConfig::from_env().unwrap_or_else(|_| DbConfig { url: "http://127.0.0.1:8000".to_string(), username: "root".to_string(), password: "root".to_string(), namespace: "fractalmind".to_string(), database: "knowledge".to_string() });

    // We won't actually connect to DB — the field type DatabaseConnection is required in AppState but won't be used.
    // For tests, we create a temporary connection by attempting to connect; if it fails, skip the test to avoid CI flakes.
    let maybe_db = match fractalmind::db::connection::connect_db(&db_cfg).await {
        Ok(db) => Some(db),
        Err(e) => {
            // Skip test gracefully if DB not available
            info!("SurrealDB not available for integration test: {}. Skipping handler test.", e);
            return;
        }
    };

    let state = AppState {
        db: maybe_db.unwrap(),
        brain,
        node_cache: NodeCache::from_env(),
        embedding_cache: EmbeddingCache::default_config(),
    };

    let shared = Arc::new(RwLock::new(state));
    let app = create_router(shared.clone()).await;

    // Build multipart body with a small text file
    let boundary = "TEST_BOUNDARY";
    let file_contents = "Hello from multipart test";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.txt\"\r\nContent-Type: text/plain\r\n\r\n{c}\r\n--{b}\r\nContent-Disposition: form-data; name=\"namespace\"\r\n\r\ntest_ns\r\n--{b}--\r\n",
        b = boundary,
        c = file_contents
    );

    let req = Request::builder()
        .method("POST")
        .uri("/v1/ingest/file")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={}", boundary))
        .body(Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.expect("router oneshot failed");
    let status = resp.status();
    assert!(status.is_success(), "Expected success status, got {}", status);

    // Convert response body to bytes (limit 64KB)
    let bytes = body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("success"));
    assert!(text.contains("node_id"));
}