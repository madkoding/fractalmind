# Fractal-Mind

Sistema de IA con memoria evolutiva y aprendizaje autónomo. Combina la precisión de grafos con la flexibilidad de embeddings vectoriales para crear un motor de conocimiento que imita la cognición humana.

## Arquitectura

### Fases de Operación

- **Vigilia (Wakefulness)**: Respuestas rápidas usando navegación fractal del grafo con SSSP
- **Fase REM (Sleep)**: Aprendizaje asíncrono, consolidación de memoria y búsqueda web

### Componentes Principales

- **Motor de Almacenamiento**: SurrealDB con persistencia en SSD e índices HNSW para vectores
- **Grafo Fractal**: Estructura recursiva RAPTOR con clustering semántico
- **Multi-Usuario**: Separación estricta con NAMESPACE (global/personal) y SCOPE por usuario
- **Navegación Optimizada**: O(m log^(2/3) n) usando hopsets para caminos mínimos

## Estructura del Proyecto

```
fractalmind/
├── src/
│   ├── models/          # Modelos de datos (nodos, aristas, embeddings)
│   ├── db/              # Configuración y queries de SurrealDB
│   ├── api/             # Endpoints HTTP (Axum)
│   ├── services/        # Lógica de negocio (ingesta, REM, queries)
│   ├── graph/           # Algoritmos de grafo (SSSP, RAPTOR)
│   ├── embeddings/      # Generación de embeddings (fastembed)
│   ├── cache/           # LRU cache para nodos frecuentes
│   └── utils/           # Utilidades generales
├── docker-compose.yml   # Orquestación de servicios
├── Dockerfile           # Imagen del backend Rust
└── FRACTAL_MIND_CORE_SPEC.md  # Especificación técnica completa
```

## Inicio Rápido

### Prerrequisitos

- Rust 1.75+
- Docker & Docker Compose
- 4GB RAM mínimo (8GB recomendado)

### Instalación

1. **Clonar y configurar**:
```bash
git clone <repo-url>
cd fractalmind

# Usar el script de desarrollo (recomendado)
./dev.sh run

# O manualmente con cargo
cargo run --features pdf
cp .env.example .env
# Editar .env con tu configuración
```

2. **Iniciar SurrealDB**:
```bash
docker-compose up -d surrealdb
```

3. **Compilar y ejecutar**:
```bash
cargo build --release
cargo run
```

O con Docker Compose completo:
```bash
docker-compose --profile full up -d
```

### Desarrollo

```bash
# Compilación rápida (desarrollo)
cargo build

# Tests
cargo test

# Tests de integración (requiere SurrealDB corriendo)
cargo test -- --ignored

# Linting
cargo clippy

# Formateo
cargo fmt
```

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/v1/ingest` | POST | Ingesta de archivos (PDF, imágenes, texto) |
| `/v1/remember` | POST | Memoria episódica/conversacional |
| `/v1/ask` | POST | Query con navegación fractal |
| `/v1/sync_rem` | POST | Trigger de fase REM (consolidación + web) |
| `/v1/memory` | PATCH | Actualización manual de conocimiento |

## Modelo de Datos

### FractalNode

Nodo fundamental del grafo con:
- **Tipo**: Leaf, Parent, Root
- **Estado**: Complete, Incomplete, Pending, Deprecated
- **Embedding**: Vector semántico (768D Nomic, 384D BGE, 512D CLIP)
- **Namespace**: global_knowledge o user_<id>
- **Metadata**: Fuente, tags, idioma, conteo de accesos

### FractalEdge

Arista con peso calculado como `1/similitud`:
- **Tipos**: ParentChild, Semantic, Temporal, CrossNamespace
- **Peso**: Para shortest path (menor = más relevante)
- **Similitud**: Coseno entre embeddings (0.0-1.0)

## Ciclo de Aprendizaje (Fase REM)

1. Detectar nodos con `status: incomplete`
2. Búsqueda web para obtener múltiples perspectivas
3. Sintetizar información en nodos nuevos (namespace global)
4. Clustering semántico de hojas → generar resúmenes padre
5. Crear enlaces entre memoria personal ↔ global

## Configuración

Variables clave en `.env`:

```bash
# Base de datos
SURREAL_URL=ws://localhost:8000
SURREAL_NS=fractalmind
SURREAL_DB=knowledge

# Embeddings
EMBEDDING_MODEL=nomic-embed-text-v1.5
EMBEDDING_DIMENSION=768

# Cache LRU
CACHE_SIZE=1000

# Fase REM
REM_INTERVAL_MINUTES=60
WEB_SEARCH_PROVIDER=searxng
WEB_SEARCH_BASE_URL=http://localhost:8080
```

## Roadmap

- [x] Modelo de datos fractal
- [x] Configuración de SurrealDB con HNSW
- [x] Estructura de proyecto Rust
- [x] Módulo LLM (Cerebro) con providers Ollama/OpenAI
- [x] Servicio de embeddings (fastembed)
- [x] Cache LRU optimizado
- [x] API REST (Axum)
- [x] Algoritmo RAPTOR para clustering
- [x] Navegación SSSP con hopsets
- [x] Fase REM asíncrona
- [x] Ingesta de PDF/imágenes con OCR
- [x] Web UI (React/Tauri)

## Licencia

MIT

<!-- AUTO-UPDATE-DATE -->
**Última actualización:** 2026-02-20 10:27:24 -03
