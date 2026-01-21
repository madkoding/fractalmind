/// Ejemplos de uso del módulo LLM (ModelBrain)
///
/// Este archivo demuestra cómo usar el cerebro del sistema para:
/// 1. Generar embeddings
/// 2. Hacer consultas de chat
/// 3. Resumir textos (fase REM)

use fractalmind::models::llm::{BrainConfig, ModelBrain};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Inicializar logging
    tracing_subscriber::fmt::init();

    println!("=== Fractal-Mind LLM Usage Examples ===\n");

    // Ejemplo 1: Configuración completamente local (Ollama)
    example_local_config().await?;

    // Ejemplo 2: Configuración híbrida (Ollama + OpenAI)
    // example_hybrid_config().await?;

    // Ejemplo 3: Uso de embeddings
    example_embeddings().await?;

    // Ejemplo 4: Chat simple
    example_simple_chat().await?;

    // Ejemplo 5: Chat con contexto
    example_chat_with_context().await?;

    // Ejemplo 6: Resumir texto (fase REM)
    example_summarize().await?;

    // Ejemplo 7: Resumir múltiples textos (RAPTOR)
    example_summarize_batch().await?;

    Ok(())
}

/// Ejemplo 1: Configuración completamente local
async fn example_local_config() -> anyhow::Result<()> {
    println!("1. Configuración Local (Ollama)");
    println!("================================\n");

    let config = BrainConfig::default_local();
    let brain = ModelBrain::new(config).await?;

    let info = brain.get_models_info();
    println!("Embedding Model: {} ({}D)", info.embedding_model, info.embedding_dimension);
    println!("Chat Model: {}", info.chat_model);
    println!("Summarizer Model: {}", info.summarizer_model);
    println!("Fully Local: {}", brain.is_fully_local());
    println!();

    Ok(())
}

/// Ejemplo 2: Configuración híbrida (comentado - requiere API key)
#[allow(dead_code)]
async fn example_hybrid_config() -> anyhow::Result<()> {
    println!("2. Configuración Híbrida (Ollama + OpenAI)");
    println!("===========================================\n");

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY no configurado");

    let config = BrainConfig::hybrid(api_key);
    let brain = ModelBrain::new(config).await?;

    let info = brain.get_models_info();
    println!("Embedding Model: {} (local)", info.embedding_model);
    println!("Chat Model: {} (remoto)", info.chat_model);
    println!("Fully Local: {}", brain.is_fully_local());
    println!();

    Ok(())
}

/// Ejemplo 3: Generar embeddings
async fn example_embeddings() -> anyhow::Result<()> {
    println!("3. Generar Embeddings");
    println!("=====================\n");

    let config = BrainConfig::default_local();
    let brain = ModelBrain::new(config).await?;

    let text = "Fractal-Mind es un sistema de IA con memoria evolutiva.";
    let response = brain.embed(text).await?;

    println!("Texto: {}", text);
    println!("Dimensión del vector: {}", response.dimension);
    println!("Primeros 5 valores: {:?}", &response.embedding[..5]);
    println!("Latencia: {}ms", response.latency_ms);
    println!("Modelo: {}", response.model);
    println!();

    Ok(())
}

/// Ejemplo 4: Chat simple
async fn example_simple_chat() -> anyhow::Result<()> {
    println!("4. Chat Simple");
    println!("==============\n");

    let config = BrainConfig::default_local();
    let brain = ModelBrain::new(config).await?;

    let question = "¿Qué es un grafo fractal?";
    let response = brain.simple_chat(question).await?;

    println!("Usuario: {}", question);
    println!("Asistente: {}", response.content);
    println!("Tokens usados: {:?}", response.tokens_used);
    println!("Latencia: {}ms", response.latency_ms);
    println!();

    Ok(())
}

/// Ejemplo 5: Chat con contexto del sistema
async fn example_chat_with_context() -> anyhow::Result<()> {
    println!("5. Chat con Contexto del Sistema");
    println!("=================================\n");

    let config = BrainConfig::default_local();
    let brain = ModelBrain::new(config).await?;

    let system_prompt = "Eres un experto en sistemas de memoria artificial y grafos de conocimiento.";
    let user_question = "Explica cómo funciona RAPTOR en 2 frases.";

    let response = brain
        .chat_with_system(system_prompt, user_question)
        .await?;

    println!("Sistema: {}", system_prompt);
    println!("Usuario: {}", user_question);
    println!("Asistente: {}", response.content);
    println!();

    Ok(())
}

/// Ejemplo 6: Resumir texto
async fn example_summarize() -> anyhow::Result<()> {
    println!("6. Resumir Texto (Fase REM)");
    println!("===========================\n");

    let config = BrainConfig::default_local();
    let brain = ModelBrain::new(config).await?;

    let long_text = r#"
    Fractal-Mind es un sistema de inteligencia artificial con memoria evolutiva que combina
    la precisión de los grafos con la flexibilidad de los embeddings vectoriales. El sistema
    opera en dos fases: Vigilia, donde responde rápidamente usando navegación fractal del grafo,
    y Fase REM, donde realiza aprendizaje asíncrono, consolidación de memoria y búsqueda web.
    La arquitectura utiliza SurrealDB con índices HNSW para búsqueda vectorial eficiente en disco,
    asegurando escalabilidad sin consumir toda la RAM. El grafo fractal se construye recursivamente
    mediante el algoritmo RAPTOR, que agrupa nodos similares y genera resúmenes jerárquicos.
    "#;

    let summary = brain.summarize(long_text).await?;

    println!("Texto original ({} caracteres):", long_text.len());
    println!("{}", long_text.trim());
    println!("\nResumen:");
    println!("{}", summary);
    println!();

    Ok(())
}

/// Ejemplo 7: Resumir múltiples textos (RAPTOR)
async fn example_summarize_batch() -> anyhow::Result<()> {
    println!("7. Resumir Batch (RAPTOR)");
    println!("=========================\n");

    let config = BrainConfig::default_local();
    let brain = ModelBrain::new(config).await?;

    let texts = vec![
        "Los embeddings vectoriales permiten representar texto como puntos en espacio multidimensional.".to_string(),
        "Los grafos de conocimiento organizan información como nodos y aristas con semántica explícita.".to_string(),
        "HNSW es un algoritmo eficiente para búsqueda de vecinos más cercanos en espacios vectoriales.".to_string(),
    ];

    println!("Textos a resumir:");
    for (i, text) in texts.iter().enumerate() {
        println!("  {}. {}", i + 1, text);
    }

    let summary = brain.summarize_batch(&texts).await?;

    println!("\nResumen jerárquico:");
    println!("{}", summary);
    println!();

    Ok(())
}
