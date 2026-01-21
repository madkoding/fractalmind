# Módulo de Modelos LLM (Cerebro)

El módulo LLM proporciona una capa de abstracción unificada para interactuar con modelos de lenguaje, tanto locales como remotos, asegurando la soberanía de los datos cuando se requiera.

## Arquitectura

### Componentes Principales

```
ModelBrain (Orquestador)
    ├── EmbeddingProvider (Trait)
    │   ├── OllamaEmbedding (Local)
    │   └── OpenAIEmbedding (Remoto)
    │
    ├── ChatProvider (Trait)
    │   ├── OllamaChat (Local)
    │   └── OpenAIChat (Remoto)
    │
    └── SummarizerProvider (Trait)
        └── OllamaSummarizer (Local)
```

### Tres Tipos de Modelos

1. **EmbeddingModel**: Vectorización de texto para búsqueda semántica
   - Ollama: `nomic-embed-text` (768D)
   - OpenAI: `text-embedding-ada-002` (1536D)

2. **ChatModel**: Respuestas finales al usuario
   - Ollama: `llama2`, `mistral`, `phi`
   - OpenAI: `gpt-4`, `gpt-3.5-turbo`

3. **SummarizerModel**: Fase REM para construir RAPTOR
   - Ollama: `llama2` con temperatura baja (0.3)
   - Optimizado para resúmenes concisos y deterministas

## Configuración

### Configuración Local (Soberanía de Datos)

```rust
use fractalmind::models::llm::{BrainConfig, ModelBrain};

let config = BrainConfig::default_local();
let brain = ModelBrain::new(config).await?;

assert!(brain.is_fully_local());  // true
```

Variables de entorno:
```bash
LLM_PREFER_LOCAL=true
EMBEDDING_PROVIDER=ollama
CHAT_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

### Configuración Híbrida

```rust
let config = BrainConfig::hybrid(openai_api_key);
let brain = ModelBrain::new(config).await?;

// Embeddings local (Ollama) + Chat remoto (OpenAI)
```

Variables de entorno:
```bash
LLM_PREFER_LOCAL=false
EMBEDDING_PROVIDER=ollama
CHAT_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Configuración desde Entorno

```rust
let config = BrainConfig::from_env()?;
let brain = ModelBrain::new(config).await?;
```

## Uso

### 1. Generar Embeddings

```rust
// Embedding simple
let response = brain.embed("Texto a vectorizar").await?;
println!("Dimensión: {}", response.dimension);
println!("Vector: {:?}", response.embedding);

// Batch embeddings
let texts = vec!["texto1".to_string(), "texto2".to_string()];
let responses = brain.embed_batch(&texts).await?;
```

### 2. Chat

```rust
// Chat simple
let response = brain.simple_chat("¿Qué es Fractal-Mind?").await?;
println!("Respuesta: {}", response.content);

// Chat con contexto del sistema
let response = brain.chat_with_system(
    "Eres un experto en IA",
    "Explica RAPTOR"
).await?;

// Chat con múltiples mensajes
use fractalmind::models::llm::traits_llm::ChatMessage;

let messages = vec![
    ChatMessage::system("Eres un asistente útil"),
    ChatMessage::user("Hola"),
    ChatMessage::assistant("¡Hola! ¿Cómo puedo ayudarte?"),
    ChatMessage::user("Explica embeddings"),
];

let response = brain.chat(&messages).await?;
```

### 3. Resumir (Fase REM)

```rust
// Resumir texto largo
let summary = brain.summarize(long_text).await?;

// Resumir múltiples textos (RAPTOR)
let texts = vec![text1, text2, text3];
let hierarchical_summary = brain.summarize_batch(&texts).await?;

// Resumir con prompt personalizado
let summary = brain.summarize_with_prompt(
    text,
    "Resume en 2 frases los puntos clave:"
).await?;
```

## Proveedores Soportados

### Ollama (Local)

**Ventajas:**
- ✅ Soberanía de datos (todo en tu máquina)
- ✅ Sin costos por API
- ✅ Sin límites de rate
- ✅ Latencia predecible

**Requisitos:**
- Ollama instalado y corriendo en `localhost:11434`
- Modelos descargados: `ollama pull nomic-embed-text`, `ollama pull llama2`

**Configuración:**
```toml
[embedding_model]
provider = { Ollama = { base_url = "http://localhost:11434", model_name = "nomic-embed-text" } }

[chat_model]
provider = { Ollama = { base_url = "http://localhost:11434", model_name = "llama2" } }
```

### OpenAI (Remoto)

**Ventajas:**
- ✅ Modelos state-of-the-art (GPT-4)
- ✅ Sin gestión de infraestructura
- ✅ Escalabilidad automática

**Desventajas:**
- ❌ Envía datos a servidores externos
- ❌ Costos por token
- ❌ Rate limits

**Configuración:**
```toml
[chat_model]
provider = { OpenAI = { api_key = "sk-...", model_name = "gpt-4" } }
```

### Extensibilidad

Para agregar un nuevo proveedor:

1. Implementa los traits:
```rust
#[async_trait]
impl EmbeddingProvider for MyProvider {
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse> {
        // Tu implementación
    }
    // ... otros métodos
}
```

2. Agrega a `ModelProvider` enum:
```rust
pub enum ModelProvider {
    // ... existentes
    MyProvider { config: MyConfig },
}
```

3. Actualiza `ModelBrain::create_*_provider`:
```rust
match &config.provider {
    // ... existentes
    ModelProvider::MyProvider { config } => {
        Ok(Arc::new(MyProvider::new(config)))
    }
}
```

## Integración con Fase REM

El `SummarizerProvider` es crítico para la fase REM:

```rust
// 1. Sistema detecta nodos incompletos
let incomplete_nodes = db.get_by_status(NodeStatus::Incomplete).await?;

// 2. Búsqueda web para cada nodo
let web_results = search_web(&incomplete_nodes).await?;

// 3. Resumir resultados web (crea conocimiento sintético)
let summaries = Vec::new();
for result in web_results {
    let summary = brain.summarize(&result.content).await?;
    summaries.push(summary);
}

// 4. Clustering de resúmenes similares
let clusters = cluster_by_similarity(&summaries).await?;

// 5. Generar resúmenes jerárquicos (RAPTOR)
for cluster in clusters {
    let parent_summary = brain.summarize_batch(&cluster).await?;
    create_parent_node(parent_summary).await?;
}
```

## Performance y Optimización

### Batch Processing

```rust
// ❌ Lento: N llamadas secuenciales
for text in texts {
    let emb = brain.embed(text).await?;
}

// ✅ Rápido: 1 llamada batch
let embeddings = brain.embed_batch(&texts).await?;
```

### Caching

Los embeddings son deterministas → cachear agresivamente:

```rust
use lru::LruCache;

struct CachedBrain {
    brain: ModelBrain,
    embedding_cache: LruCache<String, Vec<f32>>,
}

impl CachedBrain {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.embedding_cache.get(text) {
            return Ok(cached.clone());
        }

        let response = self.brain.embed(text).await?;
        self.embedding_cache.put(text.to_string(), response.embedding.clone());
        Ok(response.embedding)
    }
}
```

### Timeout y Reintentos

```rust
// Configurar timeout y reintentos
let mut config = ModelConfig::default_chat_ollama();
config.timeout_seconds = 30;
config.max_retries = 3;
```

## Health Checks

Verificar disponibilidad de proveedores:

```rust
let brain = ModelBrain::new(config).await?;

// Health checks automáticos al inicializar
// Logs: "All providers are healthy" o warnings específicos

// Checks manuales
let embedding_ok = brain.embedding_provider.health_check().await?;
let chat_ok = brain.chat_provider.health_check().await?;
let summarizer_ok = brain.summarizer_provider.health_check().await?;
```

## Soberanía de Datos

Garantizar que los datos nunca salgan de tu infraestructura:

```rust
let config = BrainConfig::default_local();
let brain = ModelBrain::new(config).await?;

// Verificar antes de procesar datos sensibles
if !brain.is_fully_local() {
    return Err(anyhow!("Data sovereignty violation: remote providers detected"));
}

// Ahora seguro procesar datos privados
let embedding = brain.embed(sensitive_text).await?;
```

## Troubleshooting

### Ollama no responde

```bash
# Verificar que Ollama está corriendo
curl http://localhost:11434/api/tags

# Ver logs
docker logs ollama  # si está en Docker
ollama serve        # si está instalado localmente

# Descargar modelo si falta
ollama pull nomic-embed-text
ollama pull llama2
```

### OpenAI rate limit

```rust
// Agregar delay entre requests
use tokio::time::{sleep, Duration};

for text in texts {
    let response = brain.embed(text).await?;
    sleep(Duration::from_millis(100)).await;
}
```

### Latencia alta

```bash
# Verificar latencia de red a Ollama
time curl -X POST http://localhost:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}'

# Verificar recursos del sistema
docker stats ollama
```

## Ejemplos Completos

Ver `examples/llm_usage.rs` para ejemplos ejecutables:

```bash
cargo run --example llm_usage
```
