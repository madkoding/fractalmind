# Guía de Uso: Sistema de Modelos Fractales

## 🚀 Inicio Rápido

### 1. Iniciar el Backend

```bash
# Terminal 1: Iniciar SurrealDB
surreal start --log trace --user root --pass root file://fractalmind.db

# Terminal 2: Iniciar el servidor Rust
cd fractalmind
cargo run --release
```

El servidor estará disponible en `http://localhost:3000`

### 2. Iniciar el Frontend

```bash
# Terminal 3: Iniciar la UI
cd fractalmind/ui
npm install
npm run dev
```

La UI estará disponible en `http://localhost:5173`

## 📚 Uso de la Interfaz

### Vista de Modelos

1. **Acceder al Gestor de Modelos**
   - Haz clic en el botón "Models" en la barra lateral
   - La interfaz cambiará de la vista de chat a la gestión de modelos

2. **Subir un Modelo GGUF**
   
   **Opción A: Drag & Drop**
   - Arrastra un archivo `.gguf` al área de upload
   - El archivo se subirá automáticamente
   
   **Opción B: Selector de Archivos**
   - Haz clic en "Select File"
   - Selecciona tu archivo `.gguf`
   - El upload comenzará inmediatamente

3. **Ver Modelos Subidos**
   - Los modelos aparecen en una lista con:
     - Estado actual (uploading, converting, ready, failed)
     - Tamaño del archivo
     - Fecha de creación
     - Arquitectura (layers, embedding dimensions, vocab size)

4. **Convertir un Modelo a Fractal**
   - Una vez que el modelo esté en estado "ready"
   - Haz clic en el botón de conversión (icono de refresh)
   - El estado cambiará a "converting"
   - La conversión se ejecuta en background

5. **Seleccionar Modelo Activo**
   - Haz clic en cualquier modelo de la lista
   - El modelo seleccionado tendrá borde morado
   - Este será el modelo usado cuando la estrategia sea "Fractal"

6. **Cambiar Estrategia de Inferencia**
   
   **Ollama (Direct)**:
   - Llamadas directas a la API de Ollama
   - Más rápido, menos control granular
   - No requiere conversión fractal
   
   **Fractal (Graph)**:
   - Navegación jerárquica por grafo fractal
   - Más control, permite exploración semántica
   - Requiere modelo convertido y seleccionado

7. **Eliminar un Modelo**
   - Haz clic en el icono de papelera (🗑️)
   - Confirma la eliminación
   - Se borrará el archivo GGUF y todos los datos relacionados

## 🔧 API REST

### Subir Modelo

```bash
curl -X POST http://localhost:3000/v1/models/upload \
  -F "file=@/path/to/model.gguf"
```

**Respuesta:**
```json
{
  "success": true,
  "model_id": "fractal_models:abc123",
  "message": "Model llama-2-7b.gguf uploaded successfully"
}
```

### Listar Modelos

```bash
curl http://localhost:3000/v1/models
```

**Respuesta:**
```json
{
  "models": [
    {
      "id": "fractal_models:abc123",
      "name": "llama-2-7b.gguf",
      "status": "ready",
      "architecture": {
        "model_type": "llama",
        "n_layers": 32,
        "embedding_dim": 4096,
        "vocab_size": 32000,
        "n_heads": 32,
        "ffn_dim": 11008
      },
      "file_size": 7000000000,
      "created_at": "2026-01-21T16:20:00Z"
    }
  ]
}
```

### Convertir Modelo

```bash
curl -X POST http://localhost:3000/v1/models/convert \
  -H "Content-Type: application/json" \
  -d '{"model_id": "fractal_models:abc123"}'
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Model conversion started for fractal_models:abc123"
}
```

### Obtener Detalles de Modelo

```bash
curl http://localhost:3000/v1/models/fractal_models:abc123
```

### Eliminar Modelo

```bash
curl -X DELETE http://localhost:3000/v1/models/fractal_models:abc123
```

### Cambiar Estrategia

```bash
# Cambiar a Ollama
curl -X PATCH http://localhost:3000/v1/config/model-strategy \
  -H "Content-Type: application/json" \
  -d '{"strategy": "ollama"}'

# Cambiar a Fractal (requiere model_id)
curl -X PATCH http://localhost:3000/v1/config/model-strategy \
  -H "Content-Type: application/json" \
  -d '{"strategy": "fractal", "model_id": "fractal_models:abc123"}'
```

## 📖 Conceptos

### ¿Qué es un Modelo Fractal?

Un modelo fractal es una representación jerárquica de un modelo de lenguaje que permite:

1. **Navegación Semántica**: Explora el espacio de parámetros del modelo siguiendo rutas de relevancia
2. **Estructura Auto-Similar**: Cada nivel del árbol contiene resúmenes coherentes del nivel inferior
3. **Búsqueda Eficiente**: Usa HNSW para encontrar rápidamente regiones relevantes
4. **Síntesis Contextual**: Combina información de múltiples niveles para respuestas más precisas

### Formato GGUF

GGUF (GPT-Generated Unified Format) es el formato usado por llama.cpp y Ollama:
- Almacena pesos del modelo cuantizados
- Incluye vocabulario y metadatos
- Optimizado para inferencia en CPU/GPU

### Proceso de Conversión

1. **Parseo**: Leer cabecera y metadatos del GGUF
2. **Extracción**: Obtener embeddings de capas del modelo
3. **Clustering**: Agrupar capas similares usando RAPTOR
4. **Jerarquía**: Crear árbol fractal con resúmenes recursivos
5. **Indexación**: Construir índice HNSW en SurrealDB

## 🎯 Casos de Uso

### 1. Comparar Modelos

Sube múltiples variantes del mismo modelo (diferentes cuantizaciones) y compara:
- Tamaño en disco
- Arquitectura
- Performance de conversión

### 2. Inferencia Híbrida

- Usa **Ollama** para respuestas rápidas en conversaciones
- Cambia a **Fractal** cuando necesites exploración profunda

### 3. Análisis de Arquitectura

Inspecciona la estructura interna de modelos:
- Número de capas
- Dimensiones de embeddings
- Tamaño del vocabulario

## ⚠️ Limitaciones Actuales

1. **Conversión Básica**: La conversión fractal extrae metadatos pero aún no genera la jerarquía RAPTOR completa
2. **Sin Progress Tracking**: No hay barra de progreso durante la conversión
3. **Storage Local**: Los archivos GGUF se almacenan en `/var/tmp/fractalmind_models`
4. **Sin Streaming**: La subida no soporta archivos parciales (usa el archivo completo)

## 🔜 Próximas Mejoras

- [ ] Implementación completa de RAPTOR para conversión
- [ ] Barra de progreso en tiempo real
- [ ] Soporte para múltiples formatos (safetensors, pytorch)
- [ ] Comparación visual entre modelos
- [ ] Export/import de modelos convertidos

## 🐛 Troubleshooting

### El modelo no se sube

- Verifica que sea un archivo `.gguf` válido
- Revisa el tamaño (archivos >50GB pueden fallar)
- Comprueba el espacio en disco de `/var/tmp`

### La conversión falla

- Mira los logs del backend: `cargo run` mostrará errores
- Verifica que SurrealDB esté corriendo
- Comprueba que el archivo GGUF no esté corrupto

### La estrategia Fractal no funciona

- Asegúrate de tener un modelo seleccionado
- El modelo debe estar en estado "ready"
- La implementación RAPTOR completa está pendiente (v0.2)

## 📞 Soporte

Para reportar bugs o solicitar features:
- Revisa los logs: `cargo run` (backend) y consola del navegador (frontend)
- Abre un issue con detalles del modelo y error
- Incluye el output de `cargo --version` y `npm --version`

---

**Versión**: 0.1.0  
**Fecha**: 2026-01-21  
**Estado**: Frontend funcional, conversión básica implementada
