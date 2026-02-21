# FractalMind

[![Release](https://img.shields.io/github/v/release/madkoding/fractalmind?logo=github)](https://github.com/madkoding/fractalmind/releases/tag/v2.0.0) [![CI](https://img.shields.io/github/actions/workflow/status/madkoding/fractalmind/ci.yml?logo=github&label=CI)](https://github.com/madkoding/fractalmind/actions) [![Rust](https://img.shields.io/badge/rust-1.75+-DC2626?logo=rust&logoColor=white)](https://www.rust-lang.org)

AI system with evolutionary memory and autonomous learning. Combines graph precision with vector embedding flexibility to create a knowledge engine that mimics human cognition.

## Architecture

### Operating Phases

- **Vigilia (Wakefulness)**: Fast responses using fractal graph navigation with SSSP
- **REM Phase (Sleep)**: Asynchronous learning, memory consolidation, and web search

### Core Components

- **Storage Engine**: SurrealDB with SSD persistence and HNSW indices for vectors
- **Fractal Graph**: RAPTOR recursive structure with semantic clustering
- **Multi-User**: Strict separation with NAMESPACE (global/personal) and SCOPE per user
- **Optimized Navigation**: O(m log^(2/3) n) using hopsets for shortest paths

## Project Structure

```
fractalmind/
├── src/
│   ├── models/          # Data models (nodes, edges, embeddings)
│   ├── db/              # SurrealDB configuration and queries
│   ├── api/             # HTTP endpoints (Axum)
│   ├── services/        # Business logic (ingestion, REM, queries)
│   ├── graph/           # Graph algorithms (SSSP, RAPTOR)
│   ├── embeddings/      # Embedding generation (fastembed)
│   ├── cache/           # LRU cache for frequent nodes
│   └── utils/           # General utilities
├── docs/                # Technical documentation
│   ├── FRACTAL_MIND_CORE_SPEC.md
│   ├── GUIA_USO_MODELOS_FRACTALES.md
│   ├── MODELOS_FRACTALES_IMPLEMENTACION.md
│   └── OpenClaw_INTEGRATION.md
├── searxng/
│   └── searxng-config.yml   # SearXNG configuration (local web search)
├── docker-compose.yml       # Service orchestration (includes SearXNG)
└── Dockerfile               # Rust backend image
```

## Quick Start

### Prerequisites

- Rust 1.75+
- Docker & Docker Compose
- Minimum 4GB RAM (8GB recommended)

### Installation

1. **Clone and run**:
```bash
git clone <repo-url>
cd fractalmind
./dev.sh run
```

**SearXNG Integration:**
- automatically included in docker-compose (profile: searxng)
- provides web search for REM phase
- runs on http://localhost:8080
- if unavailable, REM phase web search disabled (still works with local knowledge)

2. **Build and run**:
```bash
cargo build --release
cargo run
```

Or with full Docker Compose:
```bash
docker-compose --profile full up -d
```

### Development

```bash
# Fast build (development)
cargo build

# Tests
cargo test

# Integration tests (requires running SurrealDB)
cargo test -- --ignored

# Linting
cargo clippy

# Formatting
cargo fmt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/ingest` | POST | Ingest files (PDF, images, text) |
| `/v1/remember` | POST | Episodic/conversational memory |
| `/v1/ask` | POST | Query with fractal navigation |
| `/v1/sync_rem` | POST | Trigger REM phase (consolidation + web) |
| `/v1/memory` | PATCH | Manual knowledge update |

## Data Model

### FractalNode

Fundamental graph node with:
- **Type**: Leaf, Parent, Root
- **Status**: Complete, Incomplete, Pending, Deprecated
- **Embedding**: Semantic vector (768D Nomic, 384D BGE, 512D CLIP)
- **Namespace**: global_knowledge or user_<id>
- **Metadata**: Source, tags, language, access count

### FractalEdge

Edge with weight calculated as `1/similarity`:
- **Types**: ParentChild, Semantic, Temporal, CrossNamespace
- **Weight**: For shortest path (lower = more relevant)
- **Similarity**: Cosine between embeddings (0.0-1.0)

## Learning Cycle (REM Phase)

1. Detect nodes with `status: incomplete`
2. Web search to gather multiple perspectives
3. Synthesize information into new nodes (global namespace)
4. Semantic clustering of leaves → generate parent summaries
5. Create links between personal ↔ global memory

## Configuration

Key variables in `.env`:

```bash
# Database
SURREAL_URL=ws://localhost:8000
SURREAL_NS=fractalmind
SURREAL_DB=knowledge

# Embeddings
EMBEDDING_MODEL=nomic-embed-text-v1.5
EMBEDDING_DIMENSION=768

# LRU Cache
CACHE_SIZE=1000

# REM Phase
REM_INTERVAL_MINUTES=60
WEB_SEARCH_PROVIDER=searxng
WEB_SEARCH_BASE_URL=http://localhost:8080
```

## Roadmap

- [x] Fractal data model
- [x] SurrealDB configuration with HNSW
- [x] Rust project structure
- [x] LLM module (Brain) with Ollama/OpenAI providers
- [x] Embeddings service (fastembed)
- [x] Optimized LRU cache
- [x] REST API (Axum)
- [x] RAPTOR algorithm for clustering
- [x] SSSP navigation with hopsets
- [x] Asynchronous REM phase
- [x] PDF/image ingestion with OCR
- [x] Web UI (React/Tauri)

## License

MIT

<!-- AUTO-UPDATE-DATE -->
**Última actualización:** 2026-02-20 22:35:00 -03
