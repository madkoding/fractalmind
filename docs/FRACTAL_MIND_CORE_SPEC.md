Instrucción para Claude Code: "Implementa el sistema Fractal-Mind siguiendo esta especificación técnica rigurosa. El sistema debe ser una IA de memoria evolutiva capaz de auto-aprendizaje y navegación fractal de datos en disco."
1. Misión del Sistema

Construir un motor de conocimiento que imite la cognición humana: rápida en la respuesta (Vigilia) y profunda en el aprendizaje (Fase REM). Debe integrar la precisión de los grafos con la flexibilidad de los embeddings vectoriales.
2. El Motor de Almacenamiento (SurrealDB + Rust)

    Persistencia en SSD: Configurar SurrealDB con motor de almacenamiento persistente.

    Estructura Multi-Usuario: - Uso estricto de NAMESPACE para separar la memoria global de la personal.

        Cada usuario tiene un SCOPE con acceso exclusivo a su grafo privado.

    Indexación HNSW: Los vectores deben indexarse para búsqueda de proximidad directamente en disco, evitando el desbordamiento de RAM.

3. Arquitectura del Grafo Fractal (RAPTOR + SSSP)

    Construcción Recursiva: Los documentos no son piezas estáticas. Al ingresar, se dividen en hojas. La fase REM agrupa estas hojas por similitud semántica y genera un "Nodo Padre" (resumen). Este proceso se repite hasta que el grafo tiene una estructura de árbol fractal auto-similar.

    Navegación de Camino Mínimo: Implementar el algoritmo de búsqueda de ruta más corta para saltar entre la memoria personal y la global. El sistema debe tratar la "relevancia" como el peso de la arista (1/similitud).

    Optimización O(mlog2/3n): En lugar de buscar en todo el universo de datos, el sistema debe usar "hopsets" (atajos) entre clusters para reducir la latencia de respuesta en SSD.

4. Ciclo de Aprendizaje Autónomo (Fase REM)

    Detección de Vacíos: Durante la vigilia, si el usuario menciona algo que no existe en el grafo o tiene baja confianza semántica, el nodo se marca como status: incomplete.

    Exploración de la Realidad: En la fase REM (asíncrona), el sistema debe:

        Extraer los incomplete_nodes.

        Utilizar un motor de búsqueda web (ej. Tavily, Serper o Scraper avanzado) para obtener múltiples perspectivas de la "realidad".

        Sintetizar la información externa e integrarla como nuevos nodos en la capa global.

        Crear vínculos entre la memoria personal del usuario y este nuevo conocimiento global.

5. Endpoints de la API (Axum / Rust)

    POST /v1/ingest: Ingesta de archivos (PDF, Imágenes con OCR, Texto).

    POST /v1/remember: Registro de memoria episódica/conversacional inmediata.

    POST /v1/ask: Query de usuario. Ejecuta el motor de búsqueda fractal y devuelve contexto al LLM.

    POST /v1/sync_rem: Disparador del proceso de "Sueño": consolidación de memoria, búsqueda web y re-estructuración del fractal.

    PATCH /v1/memory: Actualización manual de datos para corregir o evolucionar el conocimiento.

6. Requerimientos de Código (Rust)

    Seguridad: Uso de tokio para I/O no bloqueante.

    Rendimiento: Implementar un sistema de caché LruCache para los niveles superiores del fractal (los nodos que más se consultan).

    Extensibilidad: El sistema debe permitir añadir nuevos modelos de embedding (texto o visión) sin reconstruir toda la base de datos.

