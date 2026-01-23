#![allow(unused_imports)]

mod api;
mod cache;
mod db;
mod embeddings;
mod graph;
mod models;
mod services;
mod utils;

use std::sync::Arc;
use std::net::SocketAddr;

use anyhow::Result;
use dotenv::dotenv;
use tokio::sync::RwLock;
use tower_http::cors::{CorsLayer, Any};
use http::Method;
use tower_http::trace::TraceLayer;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::db::connection::DatabaseConnection;
use crate::models::llm::{ModelBrain, BrainConfig};
use crate::cache::{NodeCache, EmbeddingCache, CacheConfig};
use crate::api::handlers::{AppState, SharedState};
use crate::api::progress::create_progress_tracker;
use crate::services::{StorageManager, UploadSessionManager, UploadCleanupJob, RemScheduler, RemSchedulerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Cargar variables de entorno
    dotenv().ok();

    // Inicializar logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "fractalmind=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Fractal-Mind Knowledge Engine...");

    // Conectar a la base de datos
    let db_config = db::connection::DbConfig::from_env()?;
    let db = db::connection::connect_db(&db_config).await?;

    // Verificar conexión
    if !db::connection::check_connection(&db).await? {
        error!("Database connection check failed");
        return Err(anyhow::anyhow!("Failed to connect to database"));
    }

    info!("Database connected successfully");

    // Inicializar schema
    db::schema::initialize_schema(&db).await?;

    // Intentar sembrar namespace global (puede fallar si ya existe)
    if let Err(e) = db::schema::seed_global_namespace(&db).await {
        warn!("Global namespace seed skipped (may already exist): {}", e);
    }

    info!("Database schema initialized");

    // Inicializar ModelBrain
    info!("Initializing Model Brain...");
    let brain_config = BrainConfig::from_env()?;
    let brain = match ModelBrain::new(brain_config).await {
        Ok(brain) => {
            info!("Model Brain initialized successfully");
            let info = brain.get_models_info();
            info!(
                "  - Embedding model: {} ({}D)",
                info.embedding_model, info.embedding_dimension
            );
            info!("  - Chat model: {}", info.chat_model);
            info!("  - Summarizer model: {}", info.summarizer_model);
            info!("  - Fully local: {}", brain.is_fully_local());
            brain
        }
        Err(e) => {
            warn!("Failed to initialize Model Brain: {}. Starting without LLM support.", e);
            warn!("Make sure Ollama is running: ollama serve");
            // Crear configuración pero sin verificar salud
            let config = BrainConfig::from_env().unwrap_or_else(|_| BrainConfig::default_local());
            ModelBrain::new_without_health_check(config)?
        }
    };

    // Crear caches desde variables de entorno o usar defaults
    let cache_config = CacheConfig::from_env();
    let node_cache = NodeCache::new(cache_config.clone());
    let embedding_cache = EmbeddingCache::new(cache_config);
    
    // Crear progress tracker para ingestion
    let progress_tracker = create_progress_tracker();
    
    // Inicializar upload manager para modelos GGUF
    info!("Initializing upload manager...");
    let storage = StorageManager::new();
    let upload_manager = UploadSessionManager::new(storage);
    if let Err(e) = upload_manager.init().await {
        warn!("Failed to initialize upload manager (non-fatal): {}", e);
    }
    let upload_manager = Arc::new(upload_manager);
    
    // Iniciar cleanup job para sesiones de upload expiradas (cada 60 minutos)
    let cleanup_job = UploadCleanupJob::new(upload_manager.clone(), 60);
    let _cleanup_handle = cleanup_job.start();
    info!("Upload cleanup job started (runs every 60 minutes)");

    // Iniciar REM Scheduler para consolidación nocturna automática
    let rem_config = RemSchedulerConfig::from_env();
    if rem_config.enabled {
        // Clonar db y brain para el scheduler
        let db_clone = db::connection::connect_db(&db_config).await?;
        let brain_clone = ModelBrain::new_without_health_check(BrainConfig::from_env().unwrap_or_else(|_| BrainConfig::default_local()))?;
        
        let rem_scheduler = Arc::new(RemScheduler::new(rem_config.clone(), db_clone, brain_clone));
        let _rem_handle = rem_scheduler.start();
        info!(
            "🌙 REM Scheduler enabled (active hours: {:02}:00 - {:02}:00)",
            rem_config.start_hour, rem_config.end_hour
        );
    } else {
        info!("REM Scheduler disabled (set REM_SCHEDULER_ENABLED=true to enable)");
    }

    // Crear estado compartido
    let state = Arc::new(RwLock::new(AppState {
        db,
        brain,
        node_cache,
        embedding_cache,
        progress_tracker,
        upload_manager,
    }));

    // Configurar CORS (permisivo para desarrollo)
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(Any)
        .expose_headers(Any)
        .max_age(std::time::Duration::from_secs(3600));

    // Crear router usando el módulo api::routes
    // IMPORTANTE: El orden de los layers es de abajo hacia arriba
    // CORS debe procesarse primero (último en la cadena de .layer())
    let app = api::routes::create_router(state)
        // Middleware (orden inverso: cors se aplica primero a las requests)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Obtener puerto desde env
    let port: u16 = std::env::var("SERVER_PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse()
        .unwrap_or(3000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    info!("Fractal-Mind API server starting on http://{}", addr);
    api::routes::print_routes();

    // Iniciar servidor
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Fractal-Mind shut down gracefully");

    Ok(())
}

/// Señal de shutdown graceful
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
    info!("Shutdown signal received...");
}
