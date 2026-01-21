#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Namespace para separación de memoria global vs personal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Namespace {
    /// Nombre del namespace
    pub name: String,

    /// Tipo de namespace
    pub namespace_type: NamespaceType,

    /// Descripción
    pub description: String,

    /// Timestamp de creación
    pub created_at: DateTime<Utc>,
}

/// Tipo de namespace
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum NamespaceType {
    /// Conocimiento global compartido
    Global,
    /// Conocimiento personal de usuario
    User,
    /// Namespace temporal/experimental
    Temporary,
}

impl Namespace {
    /// Crea el namespace global
    pub fn global() -> Self {
        Self {
            name: "global_knowledge".to_string(),
            namespace_type: NamespaceType::Global,
            description: "Shared global knowledge base".to_string(),
            created_at: Utc::now(),
        }
    }

    /// Crea un namespace para un usuario específico
    pub fn user(user_id: &str) -> Self {
        Self {
            name: format!("user_{}", user_id),
            namespace_type: NamespaceType::User,
            description: format!("Personal knowledge for user {}", user_id),
            created_at: Utc::now(),
        }
    }

    /// Crea un namespace temporal
    pub fn temporary(name: &str) -> Self {
        Self {
            name: format!("temp_{}", name),
            namespace_type: NamespaceType::Temporary,
            description: format!("Temporary namespace: {}", name),
            created_at: Utc::now(),
        }
    }
}

/// Scope de usuario para acceso exclusivo
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Scope {
    /// ID del usuario
    pub user_id: String,

    /// Namespace asociado
    pub namespace: String,

    /// Permisos del scope
    pub permissions: ScopePermissions,

    /// Timestamp de creación
    pub created_at: DateTime<Utc>,

    /// Timestamp de expiración (si aplica)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
}

/// Permisos del scope
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ScopePermissions {
    /// Puede leer nodos
    pub read: bool,

    /// Puede escribir nodos
    pub write: bool,

    /// Puede eliminar nodos
    pub delete: bool,

    /// Puede acceder al namespace global
    pub access_global: bool,
}

impl Scope {
    /// Crea un scope con permisos completos
    pub fn new_full_access(user_id: String, namespace: String) -> Self {
        Self {
            user_id,
            namespace,
            permissions: ScopePermissions {
                read: true,
                write: true,
                delete: true,
                access_global: true,
            },
            created_at: Utc::now(),
            expires_at: None,
        }
    }

    /// Crea un scope de solo lectura
    pub fn new_read_only(user_id: String, namespace: String) -> Self {
        Self {
            user_id,
            namespace,
            permissions: ScopePermissions {
                read: true,
                write: false,
                delete: false,
                access_global: true,
            },
            created_at: Utc::now(),
            expires_at: None,
        }
    }

    /// Verifica si el scope tiene permiso de escritura
    pub fn can_write(&self) -> bool {
        self.permissions.write && !self.is_expired()
    }

    /// Verifica si el scope tiene permiso de eliminación
    pub fn can_delete(&self) -> bool {
        self.permissions.delete && !self.is_expired()
    }

    /// Verifica si el scope ha expirado
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_namespace() {
        let ns = Namespace::global();
        assert_eq!(ns.name, "global_knowledge");
        assert_eq!(ns.namespace_type, NamespaceType::Global);
    }

    #[test]
    fn test_user_namespace() {
        let ns = Namespace::user("alice");
        assert_eq!(ns.name, "user_alice");
        assert_eq!(ns.namespace_type, NamespaceType::User);
    }

    #[test]
    fn test_temporary_namespace() {
        let ns = Namespace::temporary("experiment");
        assert_eq!(ns.name, "temp_experiment");
        assert_eq!(ns.namespace_type, NamespaceType::Temporary);
    }

    #[test]
    fn test_scope_full_access() {
        let scope = Scope::new_full_access(
            "alice".to_string(),
            "user_alice".to_string(),
        );

        assert!(scope.can_write());
        assert!(scope.can_delete());
        assert!(!scope.is_expired());
    }

    #[test]
    fn test_scope_read_only() {
        let scope = Scope::new_read_only(
            "bob".to_string(),
            "global_knowledge".to_string(),
        );

        assert!(!scope.can_write());
        assert!(!scope.can_delete());
        assert!(scope.permissions.read);
    }

    #[test]
    fn test_scope_expiration() {
        use chrono::Duration;

        let mut scope = Scope::new_full_access(
            "alice".to_string(),
            "user_alice".to_string(),
        );

        // Scope sin expiración
        assert!(!scope.is_expired());

        // Scope expirado en el pasado
        scope.expires_at = Some(Utc::now() - Duration::hours(1));
        assert!(scope.is_expired());
        assert!(!scope.can_write()); // No puede escribir si expiró

        // Scope que expira en el futuro
        scope.expires_at = Some(Utc::now() + Duration::hours(1));
        assert!(!scope.is_expired());
        assert!(scope.can_write());
    }
}
