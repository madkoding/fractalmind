//! Web search provider for REM phase.

#![allow(dead_code)]

use std::time::Instant;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::config::WebSearchConfig;

/// A single search result from a web search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title of the result.
    pub title: String,

    /// URL of the result.
    pub url: String,

    /// Snippet or description.
    pub snippet: String,

    /// Relevance score (0.0-1.0).
    pub score: f32,

    /// Source domain.
    pub domain: Option<String>,

    /// Published date (if available).
    pub published_date: Option<String>,
}

/// Response from a web search operation.
#[derive(Debug, Clone)]
pub struct SearchResponse {
    /// Query that was searched.
    pub query: String,

    /// Search results.
    pub results: Vec<SearchResult>,

    /// Total results available (may be more than returned).
    pub total_results: usize,

    /// Time taken in milliseconds.
    pub latency_ms: u64,

    /// Provider that served the request.
    pub provider: String,
}

impl SearchResponse {
    /// Creates a new search response.
    pub fn new(query: String, results: Vec<SearchResult>, provider: String) -> Self {
        Self {
            query,
            total_results: results.len(),
            results,
            latency_ms: 0,
            provider,
        }
    }

    /// Gets the top result.
    pub fn top_result(&self) -> Option<&SearchResult> {
        self.results.first()
    }

    /// Gets results above a score threshold.
    pub fn results_above_threshold(&self, threshold: f32) -> Vec<&SearchResult> {
        self.results.iter().filter(|r| r.score >= threshold).collect()
    }

    /// Combines snippets from all results.
    pub fn combined_snippets(&self) -> String {
        self.results
            .iter()
            .map(|r| format!("• {} ({})\n{}", r.title, r.url, r.snippet))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Trait for web search providers.
#[async_trait]
pub trait WebSearchProvider: Send + Sync {
    /// Performs a web search.
    async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse>;

    /// Checks if the provider is healthy.
    async fn health_check(&self) -> Result<bool>;

    /// Gets the provider name.
    fn provider_name(&self) -> &str;
}

/// Mock web search provider for testing.
pub struct MockSearchProvider {
    config: WebSearchConfig,
    should_fail: bool,
}

impl MockSearchProvider {
    /// Creates a new mock provider.
    pub fn new(config: WebSearchConfig) -> Self {
        Self {
            config,
            should_fail: false,
        }
    }

    /// Creates a mock provider that always fails.
    pub fn failing() -> Self {
        Self {
            config: WebSearchConfig::default(),
            should_fail: true,
        }
    }

    /// Sets whether the provider should fail.
    pub fn set_should_fail(&mut self, fail: bool) {
        self.should_fail = fail;
    }

    /// Generates mock results based on query.
    fn generate_mock_results(&self, query: &str, max_results: usize) -> Vec<SearchResult> {
        let topics = vec![
            ("Wikipedia", "https://en.wikipedia.org"),
            ("Stack Overflow", "https://stackoverflow.com"),
            ("GitHub", "https://github.com"),
            ("Documentation", "https://docs.rs"),
            ("Blog Post", "https://medium.com"),
        ];

        topics
            .into_iter()
            .take(max_results)
            .enumerate()
            .map(|(i, (title_prefix, domain))| SearchResult {
                title: format!("{} - {}", title_prefix, query),
                url: format!("{}/search?q={}", domain, query.replace(' ', "+")),
                snippet: format!(
                    "This is a mock search result for '{}'. \
                     It contains relevant information about the topic \
                     and provides context for the REM phase synthesis.",
                    query
                ),
                score: 0.9 - (i as f32 * 0.1),
                domain: Some(domain.replace("https://", "")),
                published_date: Some("2024-01-15".to_string()),
            })
            .collect()
    }
}

#[async_trait]
impl WebSearchProvider for MockSearchProvider {
    async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse> {
        let start = Instant::now();

        if self.should_fail {
            return Err(anyhow::anyhow!("Mock search provider configured to fail"));
        }

        if query.trim().is_empty() {
            return Err(anyhow::anyhow!("Query cannot be empty"));
        }

        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let max = max_results.min(self.config.max_results);
        let results = self.generate_mock_results(query, max);

        Ok(SearchResponse {
            query: query.to_string(),
            total_results: results.len(),
            results,
            latency_ms: start.elapsed().as_millis() as u64,
            provider: "mock".to_string(),
        })
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(!self.should_fail)
    }

    fn provider_name(&self) -> &str {
        "mock"
    }
}

// =============================================================================
// SearXNG Provider
// =============================================================================

/// SearXNG API response structure.
#[derive(Debug, Deserialize)]
struct SearxngApiResponse {
    results: Vec<SearxngResult>,
    #[serde(default)]
    number_of_results: usize,
}

/// Single result from SearXNG API.
#[derive(Debug, Deserialize)]
struct SearxngResult {
    title: String,
    url: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    score: Option<f32>,
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    parsed_url: Option<Vec<String>>,
    #[serde(default, rename = "publishedDate")]
    published_date: Option<String>,
}

/// SearXNG web search provider.
pub struct SearxngProvider {
    config: WebSearchConfig,
    client: Client,
    base_url: String,
}

impl SearxngProvider {
    /// Creates a new SearXNG provider.
    pub fn new(config: WebSearchConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8080".to_string());

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            base_url,
        }
    }

    /// Extracts domain from URL.
    fn extract_domain(url: &str) -> Option<String> {
        url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_string()))
    }
}

#[async_trait]
impl WebSearchProvider for SearxngProvider {
    async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse> {
        let start = Instant::now();

        if query.trim().is_empty() {
            return Err(anyhow!("Query cannot be empty"));
        }

        let max = max_results.min(self.config.max_results);

        // Build SearXNG API URL with JSON format
        let search_url = format!(
            "{}/search?q={}&format=json&pageno=1",
            self.base_url.trim_end_matches('/'),
            urlencoding::encode(query)
        );

        tracing::debug!("SearXNG search URL: {}", search_url);

        let response = self
            .client
            .get(&search_url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| anyhow!("SearXNG request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "SearXNG returned error status: {}",
                response.status()
            ));
        }

        let api_response: SearxngApiResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse SearXNG response: {}", e))?;

        // Convert SearXNG results to our format
        let results: Vec<SearchResult> = api_response
            .results
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(i, r)| {
                // Calculate score: use SearXNG score if available, otherwise decay by position
                let score = r.score.unwrap_or_else(|| 0.95 - (i as f32 * 0.05)).clamp(0.0, 1.0);

                SearchResult {
                    title: r.title,
                    url: r.url.clone(),
                    snippet: if r.content.is_empty() {
                        "No description available.".to_string()
                    } else {
                        r.content
                    },
                    score,
                    domain: Self::extract_domain(&r.url),
                    published_date: r.published_date,
                }
            })
            .collect();

        let total_results = api_response.number_of_results.max(results.len());

        Ok(SearchResponse {
            query: query.to_string(),
            total_results,
            results,
            latency_ms: start.elapsed().as_millis() as u64,
            provider: "searxng".to_string(),
        })
    }

    async fn health_check(&self) -> Result<bool> {
        let health_url = format!("{}/healthz", self.base_url.trim_end_matches('/'));

        match self.client.get(&health_url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => {
                // Try a simple search as fallback health check
                let test_url = format!(
                    "{}/search?q=test&format=json&pageno=1",
                    self.base_url.trim_end_matches('/')
                );
                match self.client.get(&test_url).send().await {
                    Ok(r) => Ok(r.status().is_success()),
                    Err(_) => Ok(false),
                }
            }
        }
    }

    fn provider_name(&self) -> &str {
        "searxng"
    }
}

// =============================================================================
// Factory
// =============================================================================

/// Factory for creating web search providers.
pub struct WebSearchFactory;

impl WebSearchFactory {
    /// Creates a web search provider based on configuration.
    pub fn create(config: WebSearchConfig) -> Box<dyn WebSearchProvider> {
        match config.provider.as_str() {
            "searxng" => Box::new(SearxngProvider::new(config)),
            "mock" => Box::new(MockSearchProvider::new(config)),
            _ => {
                tracing::warn!(
                    "Unknown search provider '{}', falling back to mock",
                    config.provider
                );
                Box::new(MockSearchProvider::new(config))
            }
        }
    }

    /// Creates a SearXNG provider with default configuration.
    pub fn searxng(base_url: &str) -> Box<dyn WebSearchProvider> {
        let config = WebSearchConfig::new("searxng").with_base_url(base_url.to_string());
        Box::new(SearxngProvider::new(config))
    }

    /// Creates a mock provider for testing.
    pub fn mock() -> Box<dyn WebSearchProvider> {
        Box::new(MockSearchProvider::new(WebSearchConfig::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_search() {
        let provider = MockSearchProvider::new(WebSearchConfig::default());
        let response = provider.search("rust programming", 3).await.unwrap();

        assert_eq!(response.query, "rust programming");
        assert_eq!(response.results.len(), 3);
        assert_eq!(response.provider, "mock");
        assert!(response.latency_ms < 1000);
    }

    #[tokio::test]
    async fn test_mock_provider_empty_query() {
        let provider = MockSearchProvider::new(WebSearchConfig::default());
        let result = provider.search("", 5).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_provider_failing() {
        let provider = MockSearchProvider::failing();
        let result = provider.search("test", 5).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_provider_health_check() {
        let provider = MockSearchProvider::new(WebSearchConfig::default());
        assert!(provider.health_check().await.unwrap());

        let failing = MockSearchProvider::failing();
        assert!(!failing.health_check().await.unwrap());
    }

    #[test]
    fn test_search_result_scores() {
        let provider = MockSearchProvider::new(WebSearchConfig::default());
        let results = provider.generate_mock_results("test", 5);

        // Scores should decrease
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[tokio::test]
    async fn test_search_response_methods() {
        let provider = MockSearchProvider::new(WebSearchConfig::default());
        let response = provider.search("test query", 5).await.unwrap();

        assert!(response.top_result().is_some());
        assert!(!response.results_above_threshold(0.5).is_empty());
        assert!(!response.combined_snippets().is_empty());
    }

    #[test]
    fn test_factory_mock() {
        let provider = WebSearchFactory::mock();
        assert_eq!(provider.provider_name(), "mock");
    }

    #[test]
    fn test_factory_unknown_falls_back() {
        let config = WebSearchConfig::new("unknown_provider");
        let provider = WebSearchFactory::create(config);
        assert_eq!(provider.provider_name(), "mock");
    }

    #[test]
    fn test_factory_searxng() {
        let config = WebSearchConfig::new("searxng")
            .with_base_url("http://localhost:8080".to_string());
        let provider = WebSearchFactory::create(config);
        assert_eq!(provider.provider_name(), "searxng");
    }

    #[test]
    fn test_searxng_factory_helper() {
        let provider = WebSearchFactory::searxng("http://localhost:8080");
        assert_eq!(provider.provider_name(), "searxng");
    }

    #[test]
    fn test_searxng_provider_creation() {
        let config = WebSearchConfig::new("searxng")
            .with_base_url("http://localhost:8080".to_string())
            .with_max_results(10);
        let provider = SearxngProvider::new(config);
        assert_eq!(provider.provider_name(), "searxng");
        assert_eq!(provider.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_searxng_default_base_url() {
        let config = WebSearchConfig::new("searxng");
        let provider = SearxngProvider::new(config);
        assert_eq!(provider.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(
            SearxngProvider::extract_domain("https://www.example.com/path"),
            Some("www.example.com".to_string())
        );
        assert_eq!(
            SearxngProvider::extract_domain("http://localhost:8080/search"),
            Some("localhost".to_string())
        );
        assert_eq!(SearxngProvider::extract_domain("not-a-url"), None);
    }

    // Integration tests - require SearXNG running on localhost:8080
    // Run with: cargo test -- --ignored

    #[tokio::test]
    #[ignore = "requires SearXNG running on localhost:8080"]
    async fn test_searxng_integration_search() {
        let provider = WebSearchFactory::searxng("http://localhost:8080");
        let response = provider.search("rust programming language", 5).await;

        match response {
            Ok(res) => {
                assert_eq!(res.provider, "searxng");
                assert!(!res.results.is_empty());
                println!("SearXNG returned {} results", res.results.len());
                for result in &res.results {
                    println!("  - {}: {}", result.title, result.url);
                }
            }
            Err(e) => {
                panic!("SearXNG search failed: {}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires SearXNG running on localhost:8080"]
    async fn test_searxng_integration_health_check() {
        let provider = WebSearchFactory::searxng("http://localhost:8080");
        let is_healthy = provider.health_check().await.unwrap();
        assert!(is_healthy, "SearXNG should be healthy");
    }
}
