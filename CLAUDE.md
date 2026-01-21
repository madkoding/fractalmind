# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fractal-Mind is an evolutionary memory AI system that mimics human cognition through a dual-phase architecture:
- **Vigilia (Wakefulness)**: Fast, real-time query responses using fractal graph navigation
- **REM Phase (Sleep)**: Asynchronous learning, memory consolidation, and knowledge integration from external sources

The system combines graph precision (SurrealDB) with vector flexibility (embeddings) to create a self-learning knowledge engine that operates on disk-based storage for scalability.

## Tech Stack

**Backend (Rust):**
- **Axum**: HTTP API framework
- **SurrealDB**: Multi-model database (graph + vector storage with HNSW indexing)
- **Tokio**: Async runtime for non-blocking I/O
- **LRU Cache**: In-memory caching for frequently accessed fractal nodes

**Core Algorithms:**
- **RAPTOR**: Recursive clustering for fractal graph construction
- **SSSP (Single-Source Shortest Path)**: Graph navigation with O(m log^(2/3) n) optimization using hopsets
- **HNSW**: Hierarchical Navigable Small World for vector similarity search

**External Integrations:**
- Web search engines (Tavily/Serper) for REM phase knowledge acquisition
- OCR for image/PDF ingestion
- Embedding models (text/vision) for semantic encoding

## Architecture

### Graph Structure

The system organizes knowledge as a **fractal tree**:
1. Documents are split into leaf nodes
2. REM phase clusters semantically similar leaves → generates parent summaries
3. Process repeats recursively → self-similar hierarchical structure
4. Navigation uses shortest-path algorithms with relevance as edge weight (1/similarity)

### Multi-User Design

```
NAMESPACE global_knowledge    # Shared public memory
NAMESPACE user_alice          # Private user scope
NAMESPACE user_bob            # Private user scope
```

Each user has a SCOPE with exclusive access to their personal graph. The system creates links between personal and global memory during REM consolidation.

### Memory States

Nodes are tagged with status:
- `complete`: High-confidence knowledge
- `incomplete`: Identified gaps requiring REM phase research

## API Endpoints

```
POST   /v1/ingest      # Ingest files (PDF, images with OCR, text)
POST   /v1/remember    # Store episodic/conversational memory
POST   /v1/ask         # Query fractal graph for LLM context retrieval
POST   /v1/sync_rem    # Trigger REM phase: consolidation + web learning
PATCH  /v1/memory      # Manual knowledge updates/corrections
```

## Development Setup

### Database Configuration

SurrealDB must use **persistent storage** (not in-memory):
```bash
surreal start --log trace --user root --pass root file://fractalmind.db
```

Enable HNSW indexing for vector search directly on disk to avoid RAM overflow.

### Building (Rust)

```bash
cargo build --release              # Production build
cargo build                        # Development build (faster linking)
cargo test                         # Run test suite
cargo clippy                       # Lint checks
cargo fmt                          # Format code
```

### Performance Optimization

- Use `tokio::spawn` for async tasks, never block the runtime
- Implement `LruCache` for top-level fractal nodes (most frequently queried)
- HNSW index parameters should balance recall vs. disk I/O

## Core Implementation Patterns

### Fractal Construction (REM Phase)

```rust
async fn consolidate_memory() {
    // 1. Detect incomplete nodes from Vigilia phase
    let incomplete = query_nodes_by_status("incomplete").await;

    // 2. Web search for external knowledge
    let external_data = search_web(incomplete).await;

    // 3. Synthesize into global namespace
    let new_nodes = create_nodes(external_data, "global_knowledge").await;

    // 4. Cluster leaves by semantic similarity
    let clusters = cluster_by_embedding_similarity(new_nodes).await;

    // 5. Generate parent summaries recursively
    let parent_nodes = generate_parent_summaries(clusters).await;

    // 6. Link personal ↔ global memory
    create_cross_namespace_edges(user_scope, global_scope).await;
}
```

### Query Execution (Vigilia Phase)

```rust
async fn answer_query(user_query: &str, user_scope: &str) {
    // 1. Embed query as vector
    let query_vector = embed_text(user_query).await;

    // 2. HNSW search in user scope
    let personal_results = hnsw_search(query_vector, user_scope).await;

    // 3. Shortest path to global knowledge if needed
    let global_results = sssp_search(query_vector, "global_knowledge", personal_results).await;

    // 4. Return context to LLM
    let context = merge_and_rank(personal_results, global_results);
    return context;
}
```

## Extensibility

The system supports pluggable embedding models without database rebuilds:
- Store model identifier with each vector: `{"vector": [...], "model": "nomic-embed-text-v1.5"}`
- Query filters vectors by model version before similarity search
- Allows gradual migration to newer/multimodal embeddings

## Key Constraints

1. **Disk-based operation**: All vector indices must use HNSW on disk (not in-memory)
2. **Non-blocking I/O**: All database/network calls use Tokio async
3. **Namespace isolation**: User data must never leak across scopes
4. **Fractal depth limit**: Implement max recursion depth to prevent infinite clustering
5. **REM phase scheduling**: Run asynchronously (background task), not blocking API responses
