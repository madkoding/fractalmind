# Ingestion con Barras de Progreso

## Descripción de los Cambios

### 1. Corregido el error de UTF-8 boundary (chunker.rs)
- **Problema**: El código intentaba hacer slicing de strings en posiciones que no eran límites de caracteres UTF-8 válidos, causando el panic.
- **Solución**: Agregada función `find_char_boundary()` que asegura que los slices siempre ocurran en límites de caracteres válidos.

### 2. Sistema de Progreso con Server-Sent Events (SSE)

#### Nuevo módulo: `src/api/progress.rs`
- Tracking de progreso en tiempo real para operaciones de ingestion
- Estados del proceso:
  - `initializing`: Preparando el proceso
  - `extracting`: Extrayendo texto del archivo (PDF/imagen/texto)
  - `chunking`: Dividiendo el texto en chunks
  - `embedding`: Generando embeddings para cada chunk
  - `persisting`: Guardando nodos en la base de datos
  - `complete`: Proceso finalizado

- Métricas rastreadas:
  - `total_chunks`: Total de chunks a procesar
  - `current_chunk`: Chunk actual siendo procesado
  - `embeddings_completed`: Número de embeddings completados
  - `nodes_persisted`: Número de nodos guardados en DB

#### Nueva ruta SSE:
```
GET /v1/ingest/progress/:session_id
```

#### Respuesta modificada de ingest_file:
Ahora retorna `IngestFileResponse` que incluye:
```json
{
  "success": true,
  "session_id": "uuid-aqui",
  "node_id": "...",
  "embedding_dimension": 768,
  "latency_ms": 1234,
  "message": "File ingested successfully (5 chunks created)"
}
```

## Cómo Usar

### Ejemplo con curl + websocat (para SSE)

#### 1. Subir archivo y obtener session_id:
```bash
curl -X POST http://localhost:3000/v1/ingest/file \
  -F "file=@documento.pdf" \
  -F "namespace=global_knowledge" \
  -F "tags=[\"importante\",\"docs\"]"
```

Respuesta:
```json
{
  "success": true,
  "session_id": "abc-123-def-456",
  "node_id": "node:xyz",
  "embedding_dimension": 768,
  "latency_ms": 5234,
  "message": "File ingested successfully (10 chunks created)"
}
```

#### 2. Conectarse al stream de progreso (en paralelo):
```bash
# Opción 1: Con curl (simple)
curl -N http://localhost:3000/v1/ingest/progress/abc-123-def-456

# Opción 2: Con websocat (mejor para SSE)
websocat -t "GET /v1/ingest/progress/abc-123-def-456" \
         -H="Accept: text/event-stream" \
         ws://localhost:3000
```

### Ejemplo con JavaScript (Frontend)

```javascript
async function uploadFileWithProgress(file) {
  // 1. Subir archivo
  const formData = new FormData();
  formData.append('file', file);
  formData.append('namespace', 'global_knowledge');
  formData.append('tags', JSON.stringify(['docs']));

  const uploadResponse = await fetch('http://localhost:3000/v1/ingest/file', {
    method: 'POST',
    body: formData
  });

  const { session_id } = await uploadResponse.json();
  console.log('Upload iniciado, session_id:', session_id);

  // 2. Conectarse al stream de progreso
  const eventSource = new EventSource(
    `http://localhost:3000/v1/ingest/progress/${session_id}`
  );

  eventSource.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    
    console.log(`Stage: ${progress.stage}`);
    console.log(`Message: ${progress.message}`);
    console.log(`Embeddings: ${progress.embeddings_completed}/${progress.total_chunks}`);
    console.log(`Nodes saved: ${progress.nodes_persisted}/${progress.total_chunks}`);

    // Actualizar barra de progreso en UI
    updateProgressBar(progress);

    // Cerrar cuando complete
    if (progress.stage === 'complete' || progress.error) {
      eventSource.close();
      if (progress.success) {
        console.log('✅ Ingestion completada!');
      } else {
        console.error('❌ Error:', progress.error);
      }
    }
  };

  eventSource.onerror = (error) => {
    console.error('Error en SSE:', error);
    eventSource.close();
  };
}

function updateProgressBar(progress) {
  // Calcular progreso general basado en la etapa
  let percentage = 0;
  
  switch (progress.stage) {
    case 'extracting':
      percentage = 10;
      break;
    case 'chunking':
      percentage = 20;
      break;
    case 'embedding':
      percentage = 20 + (progress.embeddings_completed / progress.total_chunks) * 50;
      break;
    case 'persisting':
      percentage = 70 + (progress.nodes_persisted / progress.total_chunks) * 25;
      break;
    case 'complete':
      percentage = 100;
      break;
  }
  
  // Actualizar UI
  document.querySelector('.progress-bar').style.width = `${percentage}%`;
  document.querySelector('.progress-text').textContent = progress.message || '';
}
```

### Ejemplo con Python

```python
import requests
import sseclient
import json

def upload_file_with_progress(file_path):
    # 1. Subir archivo
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'namespace': 'global_knowledge',
            'tags': json.dumps(['docs'])
        }
        
        response = requests.post(
            'http://localhost:3000/v1/ingest/file',
            files=files,
            data=data
        )
        
        result = response.json()
        session_id = result['session_id']
        print(f"Upload iniciado, session_id: {session_id}")
    
    # 2. Conectarse al stream de progreso
    progress_url = f'http://localhost:3000/v1/ingest/progress/{session_id}'
    
    response = requests.get(progress_url, stream=True)
    client = sseclient.SSEClient(response)
    
    for event in client.events():
        progress = json.loads(event.data)
        
        print(f"Stage: {progress['stage']}")
        print(f"Message: {progress.get('message', '')}")
        
        if progress['total_chunks'] > 0:
            emb_pct = (progress['embeddings_completed'] / progress['total_chunks']) * 100
            pers_pct = (progress['nodes_persisted'] / progress['total_chunks']) * 100
            print(f"Embeddings: {emb_pct:.1f}%")
            print(f"Persisted: {pers_pct:.1f}%")
        
        print("-" * 50)
        
        # Terminar cuando complete o falle
        if progress['stage'] == 'complete' or progress.get('error'):
            if progress.get('success'):
                print("✅ Ingestion completada!")
            else:
                print(f"❌ Error: {progress.get('error')}")
            break

# Uso
upload_file_with_progress('documento.pdf')
```

## Ejemplo de Flujo de Eventos SSE

```
data: {"session_id":"abc-123","stage":"extracting","message":"Extracting text from Pdf file","total_chunks":0,"current_chunk":0,"embeddings_completed":0,"nodes_persisted":0,"success":false}

data: {"session_id":"abc-123","stage":"chunking","message":"Splitting text (15234 chars)","total_chunks":0,"current_chunk":0,"embeddings_completed":0,"nodes_persisted":0,"success":false}

data: {"session_id":"abc-123","stage":"embedding","message":"Generating embeddings for 10 chunks","total_chunks":10,"current_chunk":0,"embeddings_completed":0,"nodes_persisted":0,"success":false}

data: {"session_id":"abc-123","stage":"embedding","message":"Embedding chunk 1/10","total_chunks":10,"current_chunk":1,"embeddings_completed":0,"nodes_persisted":0,"success":false}

data: {"session_id":"abc-123","stage":"persisting","message":"Saving chunk 1/10 to database","total_chunks":10,"current_chunk":1,"embeddings_completed":1,"nodes_persisted":0,"success":false}

...

data: {"session_id":"abc-123","stage":"complete","message":"Successfully processed 10 chunks","total_chunks":10,"current_chunk":10,"embeddings_completed":10,"nodes_persisted":10,"success":true}
```

## Notas Técnicas

1. **Limpieza automática**: Las sesiones de progreso se limpian automáticamente después de 30 segundos de completarse.

2. **Reconexión**: Si el cliente se desconecta del SSE, puede reconectarse usando el mismo session_id (dentro de los 30 segundos).

3. **Múltiples archivos**: Cada upload obtiene su propio session_id único, permitiendo trackear múltiples uploads simultáneamente.

4. **Compatibilidad**: SSE funciona sobre HTTP/1.1 y es compatible con todos los navegadores modernos.
