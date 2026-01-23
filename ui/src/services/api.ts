import type {
  AskRequest,
  AskResponse,
  IngestRequest,
  IngestResponse,
  RememberRequest,
  RememberResponse,
  HealthResponse,
  RemPhaseStatus,
  UploadModelResponse,
  ConvertModelRequest,
  ConvertModelResponse,
  ListModelsResponse,
  GetModelResponse,
  DeleteModelResponse,
  UpdateStrategyRequest,
  UpdateStrategyResponse,
} from '@/types';

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || `${location.protocol}//${location.hostname}:3000`;

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || DEFAULT_API_URL;
  }

  setBaseUrl(url: string): void {
    this.baseUrl = url;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // Health check
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  // Ask query with fractal navigation
  async ask(request: AskRequest): Promise<AskResponse> {
    const start = Date.now();
    const res = await this.request<AskResponse>('/v1/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
    // If backend provides latency_ms prefer it, otherwise compute client-side
    res.latency_ms = res.latency_ms || (Date.now() - start);
    return res;
  }

  // Ingest document
  async ingest(request: IngestRequest): Promise<IngestResponse> {
    return this.request<IngestResponse>('/v1/ingest', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Ingest file (multipart)
  async ingestFile(
    file: File,
    namespace: string,
    tags?: string[]
  ): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('namespace', namespace);
    if (tags) {
      formData.append('tags', JSON.stringify(tags));
    }

    const response = await fetch(`${this.baseUrl}/v1/ingest/file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // Remember (episodic memory)
  async remember(request: RememberRequest): Promise<RememberResponse> {
    return this.request<RememberResponse>('/v1/remember', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Trigger REM phase
  async triggerRem(): Promise<RemPhaseStatus> {
    return this.request<RemPhaseStatus>('/v1/sync_rem', {
      method: 'POST',
    });
  }

  // Get REM status
  async remStatus(): Promise<RemPhaseStatus> {
    return this.request<RemPhaseStatus>('/v1/rem/status');
  }

  // Stream ask (SSE)
  async *askStream(request: AskRequest): AsyncGenerator<string, void, unknown> {
    const response = await fetch(`${this.baseUrl}/v1/ask/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          yield data;
        }
      }
    }
  }

  // ============================================================================
  // Model Management API
  // ============================================================================

  // Upload GGUF model file
  async uploadModel(file: File): Promise<UploadModelResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/v1/models/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // Convert model to fractal structure
  async convertModel(request: ConvertModelRequest): Promise<ConvertModelResponse> {
    return this.request<ConvertModelResponse>(`/v1/models/${request.model_id}/convert`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // List all models
  async listModels(): Promise<ListModelsResponse> {
    return this.request<ListModelsResponse>('/v1/models');
  }

  // Get model details
  async getModel(modelId: string): Promise<GetModelResponse> {
    return this.request<GetModelResponse>(`/v1/models/${modelId}`);
  }

  // Delete model
  async deleteModel(modelId: string): Promise<DeleteModelResponse> {
    return this.request<DeleteModelResponse>(`/v1/models/${modelId}`, {
      method: 'DELETE',
    });
  }

  // Update model strategy
  async updateStrategy(request: UpdateStrategyRequest): Promise<UpdateStrategyResponse> {
    return this.request<UpdateStrategyResponse>('/v1/config/model-strategy', {
      method: 'PATCH',
      body: JSON.stringify(request),
    });
  }

  // List Ollama models
  async listOllamaModels(): Promise<import('@/types/models').ListOllamaModelsResponse> {
    return this.request<import('@/types/models').ListOllamaModelsResponse>('/v1/models/ollama');
  }

  // ============================================================================
  // Chunked Upload API
  // ============================================================================

  // Initialize chunked upload
  async initUpload(request: import('@/types/models').InitUploadRequest): Promise<import('@/types/models').InitUploadResponse> {
    return this.request<import('@/types/models').InitUploadResponse>('/v1/models/upload/init', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Upload a single chunk
  async uploadChunk(
    uploadId: string,
    chunkIndex: number,
    totalChunks: number,
    chunkData: Blob
  ): Promise<import('@/types/models').UploadChunkResponse> {
    const params = new URLSearchParams({
      chunk_index: String(chunkIndex),
    });
    
    const response = await fetch(`${this.baseUrl}/v1/models/upload/${uploadId}/chunk?${params}`, {
      method: 'POST',
      body: chunkData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to upload chunk ${chunkIndex}: ${error}`);
    }

    return response.json();
  }

  // Get upload/conversion progress
  async getProgress(uploadId: string): Promise<import('@/types/models').ProgressResponse> {
    return this.request<import('@/types/models').ProgressResponse>(`/v1/models/upload/${uploadId}/status`);
  }

  // Stream progress updates (SSE)
  async *progressStream(uploadId: string): AsyncGenerator<import('@/types/models').ProgressResponse, void, unknown> {
    const response = await fetch(`${this.baseUrl}/v1/models/upload/${uploadId}/progress`, {
      headers: {
        'Accept': 'text/event-stream',
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data);
          } catch {
            // Ignore invalid JSON
          }
        }
      }
    }
  }

  // Finalize upload
  async finalizeUpload(uploadId: string): Promise<import('@/types/models').FinalizeUploadResponse> {
    return this.request<import('@/types/models').FinalizeUploadResponse>(`/v1/models/upload/${uploadId}/finalize`, {
      method: 'POST',
    });
  }

  // Cancel upload
  async cancelUpload(uploadId: string): Promise<import('@/types/models').CancelUploadResponse> {
    return this.request<import('@/types/models').CancelUploadResponse>(`/v1/models/upload/${uploadId}/cancel`, {
      method: 'POST',
    });
  }
}

// Singleton instance
export const api = new ApiClient();

export default api;
