# FractalMind API Reference

## Base URL
```
http://localhost:9000
```

## Table of Contents

- [Health Check](#health-check)
- [Stats](#stats)
- [Knowledge Ingestion](#knowledge-ingestion)
- [Episode Memory](#episode-memory)
- [Knowledge Query](#knowledge-query)
- [Search](#search)
- [REM Phase](#rem-phase)
- [Memory Management](#memory-management)
- [Fractal Structure](#fractal-structure)
- [Model Management](#model-management)

---

## Health Check

### GET `/health`

Check service health and component status.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "service": "fractal-mind",
  "version": "2.0.0",
  "components": {
    "database": true,
    "llm": true,
    "cache": true
  }
}
```

---

## Stats

### GET `/stats`

Get system statistics.

**Response:**
```json
{
  "total_nodes": 1245,
  "total_edges": 3421,
  "namespaces": [
    {
      "name": "global_knowledge",
      "node_count": 1200,
      "edge_count": 3200
    },
    {
      "name": "user_alice",
      "node_count": 45,
      "edge_count": 221
    }
  ],
  "cache_metrics": {
    "size": 1000,
    "capacity": 1000,
    "hit_rate": 0.87,
    "hits": 870,
    "misses": 130
  },
  "llm_metrics": {
    "embedding_model": "nomic-embed-text-v1.5",
    "chat_model": "qwen2.5:0.5b",
    "summarizer_model": "qwen2.5:0.5b",
    "is_local": true
  }
}
```

---

## Knowledge Ingestion

### POST `/v1/ingest`

Ingest content into the knowledge graph. Generates embeddings automatically.

**Request:**
```json
{
  "content": "The capital of France is Paris. It has a population of about 2.2 million.",
  "source": "https://example.com/france",
  "source_type": "web",
  "namespace": "global_knowledge",
  "tags": ["geography", "france", "europe"],
  "language": "en"
}
```

**Response:**
```json
{
  "success": true,
  "node_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "embedding_dimension": 768,
  "latency_ms": 156,
  "message": "Content ingested (embedding generated) + fractal updated (3 parents)"
}
```

### POST `/v1/ingest/file`

Ingest file via multipart upload. Supports PDF, images (OCR), text.

**Request (Multipart Form):**
```
file: <binary>
namespace: global_knowledge
```

**Response:**
```json
{
  "success": true,
  "session_id": "sess_abc123",
  "node_id": "node_xyz789",
  "embedding_dimension": 768,
  "latency_ms": 2345,
  "message": "File ingested successfully"
}
```

---

## Episode Memory

### POST `/v1/remember`

Store episodic or conversational memory.

**Request:**
```json
{
  "content": "User asked about their lunch plans for Friday",
  "context": "Discussion about restaurants in downtown",
  "related_nodes": ["node_abc123", "node_def456"],
  "user_id": "alice"
}
```

**Response:**
```json
{
  "success": true,
  "node_id": "memory_xyz789",
  "message": "Memory stored in personal namespace user_alice"
}
```

---

## Knowledge Query

### POST `/v1/ask`

Query the knowledge graph with LLM-powered response.

**Request:**
```json
{
  "question": "What is the capital of France?",
  "namespace": "global_knowledge",
  "max_results": 5,
  "include_sources": true,
  "use_chat": true,
  "history": [
    {
      "role": "user",
      "content": "Tell me about France"
    },
    {
      "role": "assistant",
      "content": "France is a country in Western Europe..."
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "answer": "The capital of France is Paris, which has a population of about 2.2 million people.",
  "sources": [
    {
      "node_id": "node_abc123",
      "content": "The capital of France is Paris. It has a population of about 2.2 million.",
      "similarity": 0.92,
      "source": "https://example.com/france"
    }
  ],
  "latency_ms": 234,
  "tokens_used": 45
}
```

---

## Search

### POST `/v1/search`

Vector similarity search with optional SSSP graph navigation.

**Request:**
```json
{
  "query": "geography of France",
  "namespace": "global_knowledge",
  "limit": 10,
  "threshold": 0.6
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "node_id": "node_abc123",
      "content": "The capital of France is Paris...",
      "similarity": 0.89,
      "namespace": "global_knowledge",
      "node_type": "Leaf",
      "metadata": {
        "source": "https://example.com/france",
        "tags": ["geography", "france", "europe"]
      },
      "depth_level": 0,
      "graph_path": ["node_root1", "node_parent1", "node_abc123"]
    }
  ],
  "total": 15,
  "latency_ms": 45,
  "used_sssp": true
}
```

---

## REM Phase

### POST `/v1/sync_rem`

Trigger REM phase synchronization (memory consolidation + web search).

**Request:**
```json
{
  "namespace": "global_knowledge",
  "max_nodes": 100,
  "enable_web_search": true,
  "enable_clustering": true
}
```

**Response:**
```json
{
  "success": true,
  "nodes_processed": 87,
  "nodes_created": 12,
  "nodes_updated": 45,
  "clusters_formed": 8,
  "latency_ms": 5678,
  "message": "REM phase completed: 87 nodes processed, 12 new parent nodes created"
}
```

---

## Memory Management

### PATCH `/v1/memory`

Update an existing memory node.

**Request:**
```json
{
  "node_id": "node_abc123",
  "content": "Updated content with new information",
  "status": "complete",
  "tags": ["geography", "france", "europe", "capital"],
  "deprecated": false
}
```

**Response:**
```json
{
  "success": true,
  "node_id": "node_abc123",
  "updated_fields": ["content", "tags"],
  "message": "Node updated successfully"
}
```

---

## Fractal Structure

### POST `/v1/build-fractal`

Build or rebuild fractal hierarchy (RAPTOR) for a namespace.

**Request:**
```json
{
  "namespace": "global_knowledge",
  "generate_summaries": true,
  "similarity_threshold": 0.7,
  "max_depth": 5
}
```

**Response:**
```json
{
  "success": true,
  "parent_nodes_created": 15,
  "edges_created": 45,
  "max_depth": 3,
  "root_node_ids": ["node_root1", "node_root2"],
  "latency_ms": 1234,
  "message": "Fractal structure rebuilt: 15 parent nodes created"
}
```

---

## Model Management

### Initialize Chunked Upload

#### POST `/v1/models/upload/init`

Initialize a chunked upload for a model file.

**Request:**
```json
{
  "filename": "qwen2.5-0.5b.Q4_K_M.gguf",
  "total_size": 524288000,
  "chunk_size": 10485760
}
```

**Response:**
```json
{
  "upload_id": "upload_abc123",
  "chunk_size": 10485760,
  "total_chunks": 50
}
```

### Upload Chunk

#### POST `/v1/models/upload/:upload_id/chunk`

Upload a chunk of the model file.

**Request (Form Data):**
```
chunk: <binary data>
chunk_index: 0
upload_id: upload_abc123
```

**Response:**
```json
{
  "success": true,
  "chunk_index": 0,
  "chunks_received": 1,
  "total_chunks": 50
}
```

### Finalize Upload

#### POST `/v1/models/upload/:upload_id/finalize`

Finalize the upload and start model processing.

**Response:**
```json
{
  "success": true,
  "model_id": "model_qwen_05b",
  "message": "Upload complete, conversion started"
}
```

### Get Upload Status

#### GET `/v1/models/upload/:upload_id/status`

Check upload and conversion progress.

**Response:**
```json
{
  "upload_progress": 100.0,
  "conversion_progress": 45.0,
  "status": "converting",
  "upload_speed_mbps": 12.5,
  "chunks_received": 50,
  "total_chunks": 50,
  "current_phase": "parsing"
}
```

### List Ollama Models

#### GET `/v1/models/ollama`

List available models in Ollama.

**Response:**
```json
{
  "models": [
    {
      "name": "qwen2.5:0.5b",
      "model": "qwen2.5:0.5b-q4_K_M-bf16",
      "modified_at": "2026-02-15T10:30:00Z",
      "size": 524288000,
      "digest": "sha256:abc123...",
      "details": {
        "parameter_size": "0.5B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

### List All Models

#### GET `/v1/models`

List all models in FractalMind.

**Response:**
```json
{
  "models": [
    {
      "id": "model_qwen_05b",
      "name": "qwen2.5-0.5b",
      "status": "ready",
      "architecture": {"type": "transformer", "layers": 28},
      "file_size": 524288000,
      "created_at": "2026-02-15T10:30:00Z"
    }
  ]
}
```

### Get Model Details

#### GET `/v1/models/:model_id`

Get detailed information about a specific model.

**Response:**
```json
{
  "model": {
    "id": "model_qwen_05b",
    "name": "qwen2.5-0.5b",
    "status": "ready",
    "architecture": {"type": "transformer", "layers": 28},
    "file_size": 524288000,
    "created_at": "2026-02-15T10:30:00Z"
  }
}
```

### Delete Model

#### DELETE `/v1/models/:model_id`

Delete a model.

**Response:**
```json
{
  "success": true,
  "message": "Model deleted successfully"
}
```

### Convert Model

#### POST `/v1/models/:model_id/convert`

Start model conversion process.

**Request:**
```json
{
  "model_id": "model_qwen_05b"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Conversion started"
}
```

---

## Error Responses

All API endpoints return standard error response format:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Content cannot be empty",
    "details": {}
  }
}
```

**Error Codes:**
- `invalid_request` - Validation failure
- `embedding_error` - Embedding generation failed
- `database_error` - Database operation failed
- `llm_error` - LLM/api call failed
- `model_error` - Model operation failed
- `upload_error` - Upload operation failed

---

## Authentication

No authentication required (development mode).

For production, implement:
- API key header: `X-API-Key`
- OAuth2
- JWT tokens

---

**Last Updated:** 2026-02-20
