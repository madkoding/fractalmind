use anyhow::{Context, Result};
use surrealdb::engine::remote::http::{Client, Http};
use surrealdb::opt::auth::Root;
use surrealdb::Surreal;
use tracing::{info, warn};

/// Cliente de base de datos SurrealDB
pub type DatabaseConnection = Surreal<Client>;

/// Configuración de conexión a la base de datos
#[derive(Debug, Clone)]
pub struct DbConfig {
    pub url: String,
    pub username: String,
    pub password: String,
    pub namespace: String,
    pub database: String,
}

impl DbConfig {
    /// Carga la configuración desde variables de entorno
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            url: std::env::var("SURREAL_URL")
                .unwrap_or_else(|_| "http://localhost:8000".to_string()),
            username: std::env::var("SURREAL_USER").unwrap_or_else(|_| "root".to_string()),
            password: std::env::var("SURREAL_PASS").unwrap_or_else(|_| "root".to_string()),
            namespace: std::env::var("SURREAL_NS").unwrap_or_else(|_| "fractalmind".to_string()),
            database: std::env::var("SURREAL_DB").unwrap_or_else(|_| "knowledge".to_string()),
        })
    }
}

/// Conecta a la base de datos SurrealDB
pub async fn connect_db(config: &DbConfig) -> Result<DatabaseConnection> {
    info!("Connecting to SurrealDB at {}", config.url);

    // This client uses HTTP transport. Accept ws/wss inputs for backwards
    // compatibility and normalize to an HTTP(S) endpoint.
    let normalized_url = if config.url.starts_with("ws://") {
        warn!("SURREAL_URL uses ws:// but HTTP client is configured; normalizing to http://");
        config.url.replacen("ws://", "http://", 1)
    } else if config.url.starts_with("wss://") {
        warn!("SURREAL_URL uses wss:// but HTTP client is configured; normalizing to https://");
        config.url.replacen("wss://", "https://", 1)
    } else {
        config.url.clone()
    };

    // Surreal::new::<Http> expects host:port.
    let addr = normalized_url
        .trim_start_matches("http://")
        .trim_start_matches("https://");

    info!("Connecting to SurrealDB HTTP at {}", addr);

    // Conectar al servidor usando HTTP
    let db = Surreal::new::<Http>(addr)
        .await
        .context("Failed to connect to SurrealDB")?;

    // Autenticación
    db.signin(Root {
        username: &config.username,
        password: &config.password,
    })
    .await
    .context("Failed to authenticate with SurrealDB")?;

    // Seleccionar namespace y database
    db.use_ns(&config.namespace)
        .use_db(&config.database)
        .await
        .context("Failed to select namespace/database")?;

    info!(
        "Successfully connected to SurrealDB: {}/{}",
        config.namespace, config.database
    );

    Ok(db)
}

/// Verifica la conexión a la base de datos
pub async fn check_connection(db: &DatabaseConnection) -> Result<bool> {
    match db.health().await {
        Ok(_) => {
            info!("Database health check passed");
            Ok(true)
        }
        Err(e) => {
            warn!("Database health check failed: {}", e);
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to ensure env var tests run serially
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_db_config_from_env() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // Save current env vars
        let saved = [
            ("SURREAL_URL", std::env::var("SURREAL_URL").ok()),
            ("SURREAL_USER", std::env::var("SURREAL_USER").ok()),
            ("SURREAL_PASS", std::env::var("SURREAL_PASS").ok()),
            ("SURREAL_NS", std::env::var("SURREAL_NS").ok()),
            ("SURREAL_DB", std::env::var("SURREAL_DB").ok()),
        ];

        // Set test values
        std::env::set_var("SURREAL_URL", "http://testdb:8000");
        std::env::set_var("SURREAL_USER", "testuser");
        std::env::set_var("SURREAL_PASS", "testpass");
        std::env::set_var("SURREAL_NS", "testns");
        std::env::set_var("SURREAL_DB", "testdb");

        let config = DbConfig::from_env().unwrap();

        assert_eq!(config.url, "http://testdb:8000");
        assert_eq!(config.username, "testuser");
        assert_eq!(config.password, "testpass");
        assert_eq!(config.namespace, "testns");
        assert_eq!(config.database, "testdb");

        // Restore original env vars
        for (key, value) in saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn test_db_config_defaults() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // Save current env vars
        let saved = [
            ("SURREAL_URL", std::env::var("SURREAL_URL").ok()),
            ("SURREAL_USER", std::env::var("SURREAL_USER").ok()),
        ];

        // Ensure env vars are not set
        std::env::remove_var("SURREAL_URL");
        std::env::remove_var("SURREAL_USER");

        let config = DbConfig::from_env().unwrap();

        assert_eq!(config.url, "http://localhost:8000");
        assert_eq!(config.username, "root");

        // Restore original env vars
        for (key, value) in saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
