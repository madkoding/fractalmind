# Multi-Provider Support para Fractal-Mind

## Resumen
Se ha mejorado el sistema para soportar múltiples proveedores de embeddings y chat, incluyendo:
- **Ollama Local** (sin API key)
- **Ollama Cloud** (con API key)
- **OpenAI**
- **Anthropic**

## Cambios Realizados

### 1. Ollama Provider (`src/models/llm/providers/ollama.rs`)
- ✅ Soporta both local y cloud con API key
- ✅ Metodos `with_api_key()` para OllamaEmbedding, OllamaChat, OllamaSummarizer
- ✅ Todas las requests incluyen Authorization header cuando hay API key

### 2. Anthropic Provider (`src/models/llm/providers/anthropic.rs`)
- ✅ Nuevo provider para Anthropic embeddings (claude-embedding-001)
- ✅ Nuevo provider para Anthropic chat (claude-3-opus-20240229)
- ✅ Soporta batch embeddings y chat con API key

### 3. Configuración (`src/models/llm/config.rs`)
- ✅ `ModelProvider::Ollama` ahora tiene campo `api_key: Option<String>`
- ✅ `is_local()` verifica si `api_key` es None
- ✅ `requires_api_key()` incluye Ollama con API key
- ✅ Métodos `openai_chat()` y `anthropic_chat()`
- ✅ `from_env()` soporta `OLLAMA_API_KEY`

### 4. ModelBrain (`src/models/llm/brain.rs`)
- ✅ `create_embedding_provider()` soporta Ollama, OpenAI, Anthropic
- ✅ `create_chat_provider()` soporta Ollama, OpenAI, Anthropic
- ✅ `create_summarizer_provider()` soporta Ollama, OpenAI, Anthropic
- ✅ Detecta si es local o cloud basado en presencia de API key

### 5. Configuración de Entorno (`.env.example`)
- ✅ Agregado `OLLAMA_API_KEY` para Ollama Cloud
- ✅ Mejorado comments para incluir todos los providers
- ✅ Incluido `ANTHROPIC_API_KEY` y `ANTHROPIC_MODEL`

## Cómo Usar

### Ollama Local (por defecto)
```bash
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_PROVIDER=ollama
CHAT_PROVIDER=ollama
```

### Ollama Cloud
```bash
OLLAMA_BASE_URL=https://cloud.ollama.com
OLLAMA_API_KEY=your_api_key_here
EMBEDDING_PROVIDER=ollama
CHAT_PROVIDER=ollama
```

### OpenAI
```bash
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=text-embedding-ada-002
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
```

### Anthropic
```bash
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=claude-3-opus-20240229
EMBEDDING_PROVIDER=anthropic
CHAT_PROVIDER=anthropic
```

## Modelos Soportados

### Embeddings
- **Ollama**: `nomic-embed-text` (768D)
- **OpenAI**: `text-embedding-ada-002` (1536D), `text-embedding-3-small` (1536D), `text-embedding-3-large` (3072D)
- **Anthropic**: `claude-embedding-001` (2048D)

### Chat
- **Ollama**: `llama2`, `llama3`, `mistral`, etc.
- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`

## Ventajas
1. **Flexibilidad**: Elegir el mejor provider para cada use case
2. **Costo**: Ollama local es gratuito, cloud ofrece más capacidad
3. **Privacidad**: Ollama local mantiene datos on-premise
4. **Rendimiento**: OpenAI/Anthropic ofrecen modelos más avanzados
5. **Soberanía de datos**: Seleccionar providers según regulaciones
