//! API error handling module with structured error codes.
//!
//! Error categories:
//! - VALIDATION: Input validation errors (400)
//! - BUSINESS: Business logic errors (400/422)
//! - TECHNICAL: Technical/infrastructure errors (500)
//! - AUTH: Authentication/authorization errors (401/403)
//! - NOT_FOUND: Resource not found (404)
//! - SERVICE: Service unavailable (503)
//!
//! Error codes follow format: ERR_{CATEGORY}_{DOMAIN}_{NUMBER}
//! Example: ERR_VAL_EMB_001 = Validation error in Embedding domain, error #1

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    Validation,
    Business,
    Technical,
    Auth,
    NotFound,
    Service,
}

impl ErrorCategory {
    pub fn http_status(&self) -> StatusCode {
        match self {
            ErrorCategory::Validation => StatusCode::BAD_REQUEST,
            ErrorCategory::Business => StatusCode::UNPROCESSABLE_ENTITY,
            ErrorCategory::Auth => StatusCode::UNAUTHORIZED,
            ErrorCategory::NotFound => StatusCode::NOT_FOUND,
            ErrorCategory::Service => StatusCode::SERVICE_UNAVAILABLE,
            ErrorCategory::Technical => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn code_prefix(&self) -> &'static str {
        match self {
            ErrorCategory::Validation => "ERR_VAL",
            ErrorCategory::Business => "ERR_BIZ",
            ErrorCategory::Technical => "ERR_TEC",
            ErrorCategory::Auth => "ERR_AUT",
            ErrorCategory::NotFound => "ERR_NF",
            ErrorCategory::Service => "ERR_SVC",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorDomain {
    Llm,
    Database,
    Embedding,
    Ingestion,
    Search,
    Memory,
    Config,
    Upload,
    Rem,
    Graph,
}

impl ErrorDomain {
    pub fn code_suffix(&self) -> &'static str {
        match self {
            ErrorDomain::Llm => "LLM",
            ErrorDomain::Database => "DB",
            ErrorDomain::Embedding => "EMB",
            ErrorDomain::Ingestion => "ING",
            ErrorDomain::Search => "SCH",
            ErrorDomain::Memory => "MEM",
            ErrorDomain::Config => "CFG",
            ErrorDomain::Upload => "UPL",
            ErrorDomain::Rem => "REM",
            ErrorDomain::Graph => "GPH",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessages {
    pub en: String,
    pub es: String,
    pub pt: String,
}

impl ErrorMessages {
    pub fn new(en: &str, es: &str, pt: &str) -> Self {
        Self {
            en: en.to_string(),
            es: es.to_string(),
            pt: pt.to_string(),
        }
    }

    pub fn get(&self, lang: &str) -> &str {
        match lang {
            "es" | "spa" | "esp" => &self.es,
            "pt" | "por" => &self.pt,
            _ => &self.en,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructuredError {
    pub code: String,
    pub category: ErrorCategory,
    pub domain: ErrorDomain,
    pub index: u32,
    pub messages: ErrorMessages,
    pub technical_detail: Option<String>,
}

impl StructuredError {
    pub fn new(
        category: ErrorCategory,
        domain: ErrorDomain,
        index: u32,
        messages: ErrorMessages,
    ) -> Self {
        let code = format!(
            "{}_{}_{:03}",
            category.code_prefix(),
            domain.code_suffix(),
            index
        );
        Self {
            code,
            category,
            domain,
            index,
            messages,
            technical_detail: None,
        }
    }

    #[allow(dead_code)]
    pub fn with_technical_detail(mut self, detail: String) -> Self {
        self.technical_detail = Some(detail);
        self
    }

    #[allow(dead_code)]
    pub fn get_message(&self, lang: &str) -> String {
        self.messages.get(lang).to_string()
    }
}

pub struct ErrorCatalog {
    errors: HashMap<String, StructuredError>,
}

impl Default for ErrorCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorCatalog {
    pub fn new() -> Self {
        let mut catalog = Self {
            errors: HashMap::new(),
        };
        catalog.register_validation_errors();
        catalog.register_business_errors();
        catalog.register_technical_errors();
        catalog.register_auth_errors();
        catalog.register_not_found_errors();
        catalog.register_service_errors();
        catalog
    }

    fn register_validation_errors(&mut self) {
        self.errors.insert(
            "ERR_VAL_EMB_001".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Embedding,
                1,
                ErrorMessages::new(
                    "Question cannot be empty",
                    "La pregunta no puede estar vacía",
                    "A pergunta não pode estar vazia",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_EMB_002".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Embedding,
                2,
                ErrorMessages::new(
                    "Question text is too long (max 10000 characters)",
                    "El texto de la pregunta es demasiado largo (máximo 10000 caracteres)",
                    "O texto da pergunta é muito longo (máximo 10000 caracteres)",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_MEM_001".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Memory,
                1,
                ErrorMessages::new(
                    "Content cannot be empty",
                    "El contenido no puede estar vacío",
                    "O conteúdo não pode estar vazio",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_MEM_002".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Memory,
                2,
                ErrorMessages::new(
                    "Invalid memory type specified",
                    "Tipo de memoria inválido especificado",
                    "Tipo de memória inválido especificado",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_CFG_001".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Config,
                1,
                ErrorMessages::new(
                    "Invalid provider type",
                    "Tipo de proveedor inválido",
                    "Tipo de provedor inválido",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_SCH_001".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Search,
                1,
                ErrorMessages::new(
                    "Search query cannot be empty",
                    "La consulta de búsqueda no puede estar vacía",
                    "A consulta de busca não pode estar vazia",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_SCH_002".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Search,
                2,
                ErrorMessages::new(
                    "Invalid namespace specified",
                    "Namespace especificado inválido",
                    "Namespace especificado inválido",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_ING_001".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Ingestion,
                1,
                ErrorMessages::new(
                    "No content provided for ingestion",
                    "No se proporcionó contenido para ingestión",
                    "Nenhum conteúdo fornecido para ingestão",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_ING_002".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Ingestion,
                2,
                ErrorMessages::new(
                    "File type not supported",
                    "Tipo de archivo no soportado",
                    "Tipo de arquivo não suportado",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_UPL_001".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Upload,
                1,
                ErrorMessages::new(
                    "Uploaded file is empty",
                    "El archivo subido está vacío",
                    "O arquivo enviado está vazio",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_UPL_002".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Upload,
                2,
                ErrorMessages::new(
                    "File size exceeds maximum allowed",
                    "El tamaño del archivo excede el máximo permitido",
                    "O tamanho do arquivo excede o máximo permitido",
                ),
            ),
        );

        self.errors.insert(
            "ERR_VAL_UPL_003".to_string(),
            StructuredError::new(
                ErrorCategory::Validation,
                ErrorDomain::Upload,
                3,
                ErrorMessages::new(
                    "Missing 'file' field in multipart",
                    "Falta el campo 'file' en multipart",
                    "Falta o campo 'file' em multipart",
                ),
            ),
        );
    }

    fn register_business_errors(&mut self) {
        self.errors.insert(
            "ERR_BIZ_LLM_001".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Llm,
                1,
                ErrorMessages::new(
                    "LLM provider is not available. Please try again later.",
                    "El proveedor de LLM no está disponible. Por favor, inténtelo más tarde.",
                    "O provedor de LLM não está disponível. Por favor, tente novamente mais tarde.",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_LLM_002".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Llm,
                2,
                ErrorMessages::new(
                    "Embedding service is temporarily unavailable",
                    "El servicio de embeddings está temporalmente no disponible",
                    "O serviço de embeddings está temporariamente indisponível",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_LLM_003".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Llm,
                3,
                ErrorMessages::new(
                    "Chat generation failed. Please try again.",
                    "La generación de chat falló. Por favor, inténtelo de nuevo.",
                    "A geração de chat falhou. Por favor, tente novamente.",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_SCH_001".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Search,
                1,
                ErrorMessages::new(
                    "No relevant information found for your query",
                    "No se encontró información relevante para su consulta",
                    "Nenhuma informação relevante encontrada para sua consulta",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_SCH_002".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Search,
                2,
                ErrorMessages::new(
                    "Knowledge base is empty. Please add some content first.",
                    "La base de conocimientos está vacía. Por favor, añada contenido primero.",
                    "A base de conhecimento está vazia. Por favor, adicione conteúdo primeiro.",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_REM_001".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Rem,
                1,
                ErrorMessages::new(
                    "REM phase is already in progress",
                    "La fase REM ya está en progreso",
                    "A fase REM já está em andamento",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_REM_002".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Rem,
                2,
                ErrorMessages::new(
                    "Not enough nodes for REM consolidation",
                    "No hay suficientes nodos para consolidación REM",
                    "Não há nós suficientes para consolidação REM",
                ),
            ),
        );

        self.errors.insert(
            "ERR_BIZ_MEM_001".to_string(),
            StructuredError::new(
                ErrorCategory::Business,
                ErrorDomain::Memory,
                1,
                ErrorMessages::new(
                    "Cannot update memory: node not found",
                    "No se puede actualizar la memoria: nodo no encontrado",
                    "Não é possível atualizar a memória: nó não encontrado",
                ),
            ),
        );
    }

    fn register_technical_errors(&mut self) {
        self.errors.insert(
            "ERR_TEC_DB_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Database,
                1,
                ErrorMessages::new(
                    "Database connection failed",
                    "Error de conexión a la base de datos",
                    "Falha na conexão com o banco de dados",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_DB_002".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Database,
                2,
                ErrorMessages::new(
                    "Database query execution failed",
                    "Falló la ejecución de la consulta en la base de datos",
                    "Falha na execução da consulta no banco de dados",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_DB_003".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Database,
                3,
                ErrorMessages::new(
                    "Failed to save data to database",
                    "Error al guardar datos en la base de datos",
                    "Falha ao salvar dados no banco de dados",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_EMB_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Embedding,
                1,
                ErrorMessages::new(
                    "Failed to generate embedding vector",
                    "Error al generar el vector de embedding",
                    "Falha ao gerar o vetor de embedding",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_LLM_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Llm,
                1,
                ErrorMessages::new(
                    "LLM API request failed",
                    "Falló la solicitud a la API del LLM",
                    "Falha na solicitação à API do LLM",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_LLM_002".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Llm,
                2,
                ErrorMessages::new(
                    "Failed to parse LLM response",
                    "Error al analizar la respuesta del LLM",
                    "Falha ao analisar a resposta do LLM",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_GPH_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Graph,
                1,
                ErrorMessages::new(
                    "Graph navigation failed",
                    "Falló la navegación del grafo",
                    "Falha na navegação do grafo",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_GPH_002".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Graph,
                2,
                ErrorMessages::new(
                    "Failed to build fractal structure",
                    "Error al construir la estructura fractal",
                    "Falha ao construir a estrutura fractal",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_ING_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Ingestion,
                1,
                ErrorMessages::new(
                    "Failed to process uploaded file",
                    "Error al procesar el archivo subido",
                    "Falha ao processar o arquivo enviado",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_ING_002".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Ingestion,
                2,
                ErrorMessages::new(
                    "Failed to extract text from content",
                    "Error al extraer texto del contenido",
                    "Falha ao extrair texto do conteúdo",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_CFG_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Config,
                1,
                ErrorMessages::new(
                    "Failed to load configuration",
                    "Error al cargar la configuración",
                    "Falha ao carregar a configuração",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_CFG_002".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Config,
                2,
                ErrorMessages::new(
                    "Failed to save configuration",
                    "Error al guardar la configuración",
                    "Falha ao salvar a configuração",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_UPL_001".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Upload,
                1,
                ErrorMessages::new(
                    "File upload failed",
                    "Falló la subida del archivo",
                    "Falha no upload do arquivo",
                ),
            ),
        );

        self.errors.insert(
            "ERR_TEC_UPL_002".to_string(),
            StructuredError::new(
                ErrorCategory::Technical,
                ErrorDomain::Upload,
                2,
                ErrorMessages::new(
                    "Chunk upload failed",
                    "Falló la subida del fragmento",
                    "Falha no upload do fragmento",
                ),
            ),
        );
    }

    fn register_auth_errors(&mut self) {
        self.errors.insert(
            "ERR_AUT_CFG_001".to_string(),
            StructuredError::new(
                ErrorCategory::Auth,
                ErrorDomain::Config,
                1,
                ErrorMessages::new(
                    "Invalid API key",
                    "Clave API inválida",
                    "Chave API inválida",
                ),
            ),
        );

        self.errors.insert(
            "ERR_AUT_CFG_002".to_string(),
            StructuredError::new(
                ErrorCategory::Auth,
                ErrorDomain::Config,
                2,
                ErrorMessages::new(
                    "API key expired",
                    "Clave API expirada",
                    "Chave API expirada",
                ),
            ),
        );
    }

    fn register_not_found_errors(&mut self) {
        self.errors.insert(
            "ERR_NF_MEM_001".to_string(),
            StructuredError::new(
                ErrorCategory::NotFound,
                ErrorDomain::Memory,
                1,
                ErrorMessages::new(
                    "Memory node not found",
                    "Nodo de memoria no encontrado",
                    "Nó de memória não encontrado",
                ),
            ),
        );

        self.errors.insert(
            "ERR_NF_ING_001".to_string(),
            StructuredError::new(
                ErrorCategory::NotFound,
                ErrorDomain::Ingestion,
                1,
                ErrorMessages::new(
                    "Ingested content not found",
                    "Contenido ingestado no encontrado",
                    "Conteúdo ingerido não encontrado",
                ),
            ),
        );

        self.errors.insert(
            "ERR_NF_UPL_001".to_string(),
            StructuredError::new(
                ErrorCategory::NotFound,
                ErrorDomain::Upload,
                1,
                ErrorMessages::new(
                    "Upload session not found",
                    "Sesión de subida no encontrada",
                    "Sessão de upload não encontrada",
                ),
            ),
        );
    }

    fn register_service_errors(&mut self) {
        self.errors.insert(
            "ERR_SVC_LLM_001".to_string(),
            StructuredError::new(
                ErrorCategory::Service,
                ErrorDomain::Llm,
                1,
                ErrorMessages::new(
                    "LLM service is temporarily unavailable",
                    "El servicio de LLM está temporalmente no disponible",
                    "O serviço de LLM está temporariamente indisponível",
                ),
            ),
        );

        self.errors.insert(
            "ERR_SVC_EMB_001".to_string(),
            StructuredError::new(
                ErrorCategory::Service,
                ErrorDomain::Embedding,
                1,
                ErrorMessages::new(
                    "Embedding service is temporarily unavailable",
                    "El servicio de embeddings está temporalmente no disponible",
                    "O serviço de embeddings está temporariamente indisponível",
                ),
            ),
        );

        self.errors.insert(
            "ERR_SVC_DB_001".to_string(),
            StructuredError::new(
                ErrorCategory::Service,
                ErrorDomain::Database,
                1,
                ErrorMessages::new(
                    "Database service is temporarily unavailable",
                    "El servicio de base de datos está temporalmente no disponible",
                    "O serviço de banco de dados está temporariamente indisponível",
                ),
            ),
        );
    }

    pub fn get(&self, code: &str) -> Option<&StructuredError> {
        self.errors.get(code)
    }

    #[allow(dead_code)]
    pub fn register(&mut self, error: StructuredError) {
        self.errors.insert(error.code.clone(), error);
    }
}

lazy_static::lazy_static! {
    pub static ref ERROR_CATALOG: ErrorCatalog = ErrorCatalog::new();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub success: bool,
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    pub category: String,
    pub message: String,
    pub technical_detail: Option<String>,
}

impl ErrorResponse {
    pub fn new(code: &str, lang: &str, technical_detail: Option<String>) -> Self {
        if let Some(structured) = ERROR_CATALOG.get(code) {
            let category = match structured.category {
                ErrorCategory::Validation => "VALIDATION",
                ErrorCategory::Business => "BUSINESS",
                ErrorCategory::Technical => "TECHNICAL",
                ErrorCategory::Auth => "AUTH",
                ErrorCategory::NotFound => "NOT_FOUND",
                ErrorCategory::Service => "SERVICE_UNAVAILABLE",
            };
            Self {
                success: false,
                error: ErrorDetail {
                    code: code.to_string(),
                    category: category.to_string(),
                    message: structured.messages.get(lang).to_string(),
                    technical_detail,
                },
            }
        } else {
            Self {
                success: false,
                error: ErrorDetail {
                    code: code.to_string(),
                    category: "TECHNICAL".to_string(),
                    message: format!("Unknown error: {}", code),
                    technical_detail,
                },
            }
        }
    }

    #[allow(dead_code)]
    pub fn from_structured_error(error: &StructuredError, lang: &str) -> Self {
        let category = match error.category {
            ErrorCategory::Validation => "VALIDATION",
            ErrorCategory::Business => "BUSINESS",
            ErrorCategory::Technical => "TECHNICAL",
            ErrorCategory::Auth => "AUTH",
            ErrorCategory::NotFound => "NOT_FOUND",
            ErrorCategory::Service => "SERVICE_UNAVAILABLE",
        };
        Self {
            success: false,
            error: ErrorDetail {
                code: error.code.clone(),
                category: category.to_string(),
                message: error.messages.get(lang).to_string(),
                technical_detail: error.technical_detail.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApiErrorCode {
    ValidationEmptyQuestion,
    ValidationQuestionTooLong,
    ValidationEmptyContent,
    ValidationInvalidMemoryType,
    ValidationInvalidProviderType,
    ValidationEmptySearchQuery,
    ValidationInvalidNamespace,
    ValidationNoContent,
    ValidationUnsupportedFileType,
    ValidationEmptyFile,
    ValidationFileSizeExceeded,
    ValidationMissingFileField,

    BusinessLlmUnavailable,
    BusinessEmbeddingUnavailable,
    BusinessChatFailed,
    BusinessNoSearchResults,
    BusinessEmptyKnowledgeBase,
    BusinessRemInProgress,
    BusinessRemNotEnoughNodes,
    BusinessMemoryUpdateNotFound,

    TechnicalDbConnection,
    TechnicalDbQuery,
    TechnicalDbSave,
    TechnicalEmbeddingGenerate,
    TechnicalLlmRequest,
    TechnicalLlmParse,
    TechnicalGraphNavigation,
    TechnicalGraphBuild,
    TechnicalIngestionProcess,
    TechnicalIngestionExtract,
    TechnicalConfigLoad,
    TechnicalConfigSave,
    TechnicalUploadFailed,
    TechnicalChunkUpload,

    AuthInvalidApiKey,
    AuthApiKeyExpired,

    NotFoundMemoryNode,
    NotFoundIngestedContent,
    NotFoundUploadSession,

    ServiceLlmUnavailable,
    ServiceEmbeddingUnavailable,
    ServiceDatabaseUnavailable,

    Unknown,
}

impl std::fmt::Display for ApiErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code_string())
    }
}

impl ApiErrorCode {
    pub fn code_string(&self) -> &'static str {
        match self {
            ApiErrorCode::ValidationEmptyQuestion => "ERR_VAL_EMB_001",
            ApiErrorCode::ValidationQuestionTooLong => "ERR_VAL_EMB_002",
            ApiErrorCode::ValidationEmptyContent => "ERR_VAL_MEM_001",
            ApiErrorCode::ValidationInvalidMemoryType => "ERR_VAL_MEM_002",
            ApiErrorCode::ValidationInvalidProviderType => "ERR_VAL_CFG_001",
            ApiErrorCode::ValidationEmptySearchQuery => "ERR_VAL_SCH_001",
            ApiErrorCode::ValidationInvalidNamespace => "ERR_VAL_SCH_002",
            ApiErrorCode::ValidationNoContent => "ERR_VAL_ING_001",
            ApiErrorCode::ValidationUnsupportedFileType => "ERR_VAL_ING_002",
            ApiErrorCode::ValidationEmptyFile => "ERR_VAL_UPL_001",
            ApiErrorCode::ValidationFileSizeExceeded => "ERR_VAL_UPL_002",
            ApiErrorCode::ValidationMissingFileField => "ERR_VAL_UPL_003",

            ApiErrorCode::BusinessLlmUnavailable => "ERR_BIZ_LLM_001",
            ApiErrorCode::BusinessEmbeddingUnavailable => "ERR_BIZ_LLM_002",
            ApiErrorCode::BusinessChatFailed => "ERR_BIZ_LLM_003",
            ApiErrorCode::BusinessNoSearchResults => "ERR_BIZ_SCH_001",
            ApiErrorCode::BusinessEmptyKnowledgeBase => "ERR_BIZ_SCH_002",
            ApiErrorCode::BusinessRemInProgress => "ERR_BIZ_REM_001",
            ApiErrorCode::BusinessRemNotEnoughNodes => "ERR_BIZ_REM_002",
            ApiErrorCode::BusinessMemoryUpdateNotFound => "ERR_BIZ_MEM_001",

            ApiErrorCode::TechnicalDbConnection => "ERR_TEC_DB_001",
            ApiErrorCode::TechnicalDbQuery => "ERR_TEC_DB_002",
            ApiErrorCode::TechnicalDbSave => "ERR_TEC_DB_003",
            ApiErrorCode::TechnicalEmbeddingGenerate => "ERR_TEC_EMB_001",
            ApiErrorCode::TechnicalLlmRequest => "ERR_TEC_LLM_001",
            ApiErrorCode::TechnicalLlmParse => "ERR_TEC_LLM_002",
            ApiErrorCode::TechnicalGraphNavigation => "ERR_TEC_GPH_001",
            ApiErrorCode::TechnicalGraphBuild => "ERR_TEC_GPH_002",
            ApiErrorCode::TechnicalIngestionProcess => "ERR_TEC_ING_001",
            ApiErrorCode::TechnicalIngestionExtract => "ERR_TEC_ING_002",
            ApiErrorCode::TechnicalConfigLoad => "ERR_TEC_CFG_001",
            ApiErrorCode::TechnicalConfigSave => "ERR_TEC_CFG_002",
            ApiErrorCode::TechnicalUploadFailed => "ERR_TEC_UPL_001",
            ApiErrorCode::TechnicalChunkUpload => "ERR_TEC_UPL_002",

            ApiErrorCode::AuthInvalidApiKey => "ERR_AUT_CFG_001",
            ApiErrorCode::AuthApiKeyExpired => "ERR_AUT_CFG_002",

            ApiErrorCode::NotFoundMemoryNode => "ERR_NF_MEM_001",
            ApiErrorCode::NotFoundIngestedContent => "ERR_NF_ING_001",
            ApiErrorCode::NotFoundUploadSession => "ERR_NF_UPL_001",

            ApiErrorCode::ServiceLlmUnavailable => "ERR_SVC_LLM_001",
            ApiErrorCode::ServiceEmbeddingUnavailable => "ERR_SVC_EMB_001",
            ApiErrorCode::ServiceDatabaseUnavailable => "ERR_SVC_DB_001",

            ApiErrorCode::Unknown => "ERR_UNKNOWN",
        }
    }

    pub fn category(&self) -> ErrorCategory {
        match self {
            ApiErrorCode::ValidationEmptyQuestion
            | ApiErrorCode::ValidationQuestionTooLong
            | ApiErrorCode::ValidationEmptyContent
            | ApiErrorCode::ValidationInvalidMemoryType
            | ApiErrorCode::ValidationInvalidProviderType
            | ApiErrorCode::ValidationEmptySearchQuery
            | ApiErrorCode::ValidationInvalidNamespace
            | ApiErrorCode::ValidationNoContent
            | ApiErrorCode::ValidationUnsupportedFileType
            | ApiErrorCode::ValidationEmptyFile
            | ApiErrorCode::ValidationFileSizeExceeded
            | ApiErrorCode::ValidationMissingFileField => ErrorCategory::Validation,

            ApiErrorCode::BusinessLlmUnavailable
            | ApiErrorCode::BusinessEmbeddingUnavailable
            | ApiErrorCode::BusinessChatFailed
            | ApiErrorCode::BusinessNoSearchResults
            | ApiErrorCode::BusinessEmptyKnowledgeBase
            | ApiErrorCode::BusinessRemInProgress
            | ApiErrorCode::BusinessRemNotEnoughNodes
            | ApiErrorCode::BusinessMemoryUpdateNotFound => ErrorCategory::Business,

            ApiErrorCode::TechnicalDbConnection
            | ApiErrorCode::TechnicalDbQuery
            | ApiErrorCode::TechnicalDbSave
            | ApiErrorCode::TechnicalEmbeddingGenerate
            | ApiErrorCode::TechnicalLlmRequest
            | ApiErrorCode::TechnicalLlmParse
            | ApiErrorCode::TechnicalGraphNavigation
            | ApiErrorCode::TechnicalGraphBuild
            | ApiErrorCode::TechnicalIngestionProcess
            | ApiErrorCode::TechnicalIngestionExtract
            | ApiErrorCode::TechnicalConfigLoad
            | ApiErrorCode::TechnicalConfigSave
            | ApiErrorCode::TechnicalUploadFailed
            | ApiErrorCode::TechnicalChunkUpload => ErrorCategory::Technical,

            ApiErrorCode::AuthInvalidApiKey | ApiErrorCode::AuthApiKeyExpired => {
                ErrorCategory::Auth
            }

            ApiErrorCode::NotFoundMemoryNode
            | ApiErrorCode::NotFoundIngestedContent
            | ApiErrorCode::NotFoundUploadSession => ErrorCategory::NotFound,

            ApiErrorCode::ServiceLlmUnavailable
            | ApiErrorCode::ServiceEmbeddingUnavailable
            | ApiErrorCode::ServiceDatabaseUnavailable => ErrorCategory::Service,

            ApiErrorCode::Unknown => ErrorCategory::Technical,
        }
    }

    pub fn http_status(&self) -> StatusCode {
        self.category().http_status()
    }
}

#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Validation error: {0}")]
    ValidationError(ApiErrorCode),

    #[error("Business error: {0}")]
    BusinessError(ApiErrorCode),

    #[error("Technical error: {0}")]
    TechnicalError(ApiErrorCode),

    #[error("Auth error: {0}")]
    AuthError(ApiErrorCode),

    #[error("Not found: {0}")]
    NotFoundError(ApiErrorCode),

    #[error("Service error: {0}")]
    ServiceError(ApiErrorCode),

    #[error("Unknown error: {0}")]
    UnknownError(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

impl ApiError {
    #[allow(dead_code)]
    pub fn code(&self) -> ApiErrorCode {
        match self {
            ApiError::ValidationError(code) => *code,
            ApiError::BusinessError(code) => *code,
            ApiError::TechnicalError(code) => *code,
            ApiError::AuthError(code) => *code,
            ApiError::NotFoundError(code) => *code,
            ApiError::ServiceError(code) => *code,
            ApiError::UnknownError(_) => ApiErrorCode::Unknown,
            ApiError::BadRequest(_) => ApiErrorCode::ValidationEmptyQuestion,
            ApiError::InternalError(_) => ApiErrorCode::TechnicalDbQuery,
            ApiError::DatabaseError(_) => ApiErrorCode::TechnicalDbQuery,
            ApiError::EmbeddingError(_) => ApiErrorCode::TechnicalEmbeddingGenerate,
            ApiError::LlmError(_) => ApiErrorCode::TechnicalLlmRequest,
            ApiError::ServiceUnavailable(_) => ApiErrorCode::ServiceDatabaseUnavailable,
        }
    }

    #[allow(dead_code)]
    pub fn code_string(&self) -> String {
        self.code().code_string().to_string()
    }

    pub fn to_response(&self, _lang: &str) -> ErrorResponse {
        let (code, technical_detail) = self.to_error_response();
        ErrorResponse::new(&code, "en", technical_detail)
    }
}

impl ApiError {
    pub fn with_technical_detail(self, _detail: String) -> Self {
        match self {
            ApiError::ValidationError(code) => ApiError::ValidationError(code),
            ApiError::BusinessError(code) => ApiError::BusinessError(code),
            ApiError::TechnicalError(code) => ApiError::TechnicalError(code),
            ApiError::AuthError(code) => ApiError::AuthError(code),
            ApiError::NotFoundError(code) => ApiError::NotFoundError(code),
            ApiError::ServiceError(code) => ApiError::ServiceError(code),
            ApiError::UnknownError(msg) => ApiError::UnknownError(msg),
            ApiError::BadRequest(msg) => ApiError::BadRequest(msg),
            ApiError::InternalError(msg) => ApiError::InternalError(msg),
            ApiError::DatabaseError(msg) => ApiError::DatabaseError(msg),
            ApiError::EmbeddingError(msg) => ApiError::EmbeddingError(msg),
            ApiError::LlmError(msg) => ApiError::LlmError(msg),
            ApiError::ServiceUnavailable(msg) => ApiError::ServiceUnavailable(msg),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::ValidationError(_) => StatusCode::BAD_REQUEST,
            ApiError::BusinessError(code) => code.http_status(),
            ApiError::TechnicalError(code) => code.http_status(),
            ApiError::AuthError(_) => StatusCode::UNAUTHORIZED,
            ApiError::NotFoundError(_) => StatusCode::NOT_FOUND,
            ApiError::ServiceError(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::UnknownError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::DatabaseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::EmbeddingError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::LlmError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        };

        let lang = self.detect_lang();
        (status, Json(self.to_response(&lang))).into_response()
    }
}

impl ApiError {
    fn detect_lang(&self) -> String {
        std::env::var("LANG").unwrap_or_else(|_| "en".to_string())
    }

    pub fn to_error_response(&self) -> (String, Option<String>) {
        let code = match self {
            ApiError::ValidationError(code) => code.code_string(),
            ApiError::BusinessError(code) => code.code_string(),
            ApiError::TechnicalError(code) => code.code_string(),
            ApiError::AuthError(code) => code.code_string(),
            ApiError::NotFoundError(code) => code.code_string(),
            ApiError::ServiceError(code) => code.code_string(),
            ApiError::UnknownError(msg) => return ("ERR_UNKNOWN".to_string(), Some(msg.clone())),
            ApiError::BadRequest(msg) => return ("ERR_VAL_UNKNOWN".to_string(), Some(msg.clone())),
            ApiError::InternalError(msg) => {
                return ("ERR_TEC_UNKNOWN".to_string(), Some(msg.clone()))
            }
            ApiError::DatabaseError(msg) => {
                return ("ERR_TEC_DB_UNKNOWN".to_string(), Some(msg.clone()))
            }
            ApiError::EmbeddingError(msg) => {
                return ("ERR_TEC_EMB_UNKNOWN".to_string(), Some(msg.clone()))
            }
            ApiError::LlmError(msg) => {
                return ("ERR_TEC_LLM_UNKNOWN".to_string(), Some(msg.clone()))
            }
            ApiError::ServiceUnavailable(msg) => {
                return ("ERR_SVC_UNKNOWN".to_string(), Some(msg.clone()))
            }
        };
        (code.to_string(), None)
    }
}

impl From<String> for ApiError {
    fn from(s: String) -> Self {
        ApiError::UnknownError(s)
    }
}

impl From<&str> for ApiError {
    fn from(s: &str) -> Self {
        ApiError::UnknownError(s.to_string())
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        tracing::error!("Internal error: {}", err);
        ApiError::TechnicalError(ApiErrorCode::TechnicalDbQuery)
            .with_technical_detail(err.to_string())
    }
}

impl From<tokio::io::Error> for ApiError {
    fn from(err: tokio::io::Error) -> Self {
        tracing::error!("IO error: {}", err);
        ApiError::TechnicalError(ApiErrorCode::TechnicalDbQuery)
            .with_technical_detail(err.to_string())
    }
}

impl From<axum::extract::multipart::MultipartError> for ApiError {
    fn from(err: axum::extract::multipart::MultipartError) -> Self {
        ApiError::BadRequest(err.to_string())
    }
}

impl From<surrealdb::Error> for ApiError {
    fn from(err: surrealdb::Error) -> Self {
        tracing::error!("Database error: {}", err);
        ApiError::TechnicalError(ApiErrorCode::TechnicalDbQuery)
            .with_technical_detail(err.to_string())
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(
            ApiErrorCode::ValidationEmptyQuestion.code_string(),
            "ERR_VAL_EMB_001"
        );
        assert_eq!(
            ApiErrorCode::BusinessLlmUnavailable.code_string(),
            "ERR_BIZ_LLM_001"
        );
        assert_eq!(
            ApiErrorCode::TechnicalDbConnection.code_string(),
            "ERR_TEC_DB_001"
        );
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(
            ApiErrorCode::ValidationEmptyQuestion.category(),
            ErrorCategory::Validation
        );
        assert_eq!(
            ApiErrorCode::BusinessLlmUnavailable.category(),
            ErrorCategory::Business
        );
        assert_eq!(
            ApiErrorCode::TechnicalDbConnection.category(),
            ErrorCategory::Technical
        );
    }

    #[test]
    fn test_error_messages() {
        let err = ApiError::ValidationError(ApiErrorCode::ValidationEmptyQuestion);
        let response = err.to_response("es");
        assert_eq!(response.error.code, "ERR_VAL_EMB_001");
        assert_eq!(response.error.message, "Question cannot be empty");
    }

    #[test]
    fn test_error_messages_english() {
        let err = ApiError::ValidationError(ApiErrorCode::ValidationEmptyQuestion);
        let response = err.to_response("en");
        assert_eq!(response.error.message, "Question cannot be empty");
    }

    #[test]
    fn test_http_status() {
        assert_eq!(
            ApiErrorCode::ValidationEmptyQuestion.http_status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ApiErrorCode::NotFoundMemoryNode.http_status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiErrorCode::ServiceLlmUnavailable.http_status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
