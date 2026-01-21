use fractalmind::services::ingestion::{IngestionInput, IngestionService};
use fractalmind::models::{EmbeddingVector, EmbeddingModel};

#[tokio::test]
async fn test_ingest_service_from_text_file() {
    // Load fixture
    let data = tokio::fs::read("tests/fixtures/samples/sample.txt").await.unwrap();

    // Prepare input
    let input = IngestionInput::new(data, "test_namespace").with_filename("sample.txt");

    // Create service
    let service = IngestionService::with_defaults();

    // Mock embedding generator - returns constant vector
    let embedding_gen = |_: &str| -> EmbeddingVector {
        EmbeddingVector::new(vec![0.01f32; EmbeddingModel::NomicEmbedTextV15.dimension()], EmbeddingModel::NomicEmbedTextV15)
    };

    let result = service.ingest(input, embedding_gen).await.expect("ingest failed");

    assert!(result.is_successful());
    assert!(result.node_count() >= 1, "expected at least 1 generated node");

    // Verify extraction text contains snippet from fixture
    assert!(result.extraction.text.contains("Hello Fractal Mind"));
}

#[test]
fn test_ingest_text_direct() {
    let service = IngestionService::with_defaults();

    // simple short text
    let text = "Short test text";

    let embedding_gen = |_: &str| -> EmbeddingVector {
        EmbeddingVector::new(vec![0.1f32; EmbeddingModel::NomicEmbedTextV15.dimension()], EmbeddingModel::NomicEmbedTextV15)
    };

    let res = service.ingest_text(text, "ns", Some("source"), vec!["tag1".to_string()], embedding_gen);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.is_successful());
    assert_eq!(out.node_count(), 1);
    assert!(out.extraction.text.contains("Short test text"));
}