//! API error handling module.

#![allow(dead_code)]

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

/// API error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal server error: {0}")]
    InternalError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Embedding generation failed: {0}")]
    EmbeddingError(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

/// Error response body
#[derive(Serialize)]
pub struct ErrorResponse {
    pub success: bool,
    pub error: String,
    pub code: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "BAD_REQUEST"),
            ApiError::NotFound(_) => (StatusCode::NOT_FOUND, "NOT_FOUND"),
            ApiError::InternalError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
            ApiError::DatabaseError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "DATABASE_ERROR"),
            ApiError::EmbeddingError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "EMBEDDING_ERROR"),
            ApiError::LlmError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "LLM_ERROR"),
            ApiError::ValidationError(_) => (StatusCode::BAD_REQUEST, "VALIDATION_ERROR"),
            ApiError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "SERVICE_UNAVAILABLE"),
        };

        let body = ErrorResponse {
            success: false,
            error: self.to_string(),
            code: code.to_string(),
        };

        (status, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::InternalError(err.to_string())
    }
}

impl From<tokio::io::Error> for ApiError {
    fn from(err: tokio::io::Error) -> Self {
        ApiError::InternalError(err.to_string())
    }
}

impl From<axum::extract::multipart::MultipartError> for ApiError {
    fn from(err: axum::extract::multipart::MultipartError) -> Self {
        ApiError::BadRequest(err.to_string())
    }
}

/// Result type for API handlers
pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        let err = ApiError::BadRequest("Invalid input".to_string());
        assert!(err.to_string().contains("Invalid input"));

        let err = ApiError::NotFound("Resource not found".to_string());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_error_response() {
        let response = ErrorResponse {
            success: false,
            error: "Test error".to_string(),
            code: "TEST_ERROR".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("error"));
        assert!(json.contains("code"));
    }
}
