import type {
  AskRequest,
  AskResponse,
  IngestRequest,
  IngestResponse,
  RememberRequest,
  RememberResponse,
  HealthResponse,
  RemPhaseStatus,
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
    return this.request<AskResponse>('/v1/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
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
}

// Singleton instance
export const api = new ApiClient();

export default api;
