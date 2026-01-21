// API Types matching Rust backend

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface AskRequest {
  question: string;
  namespace?: string;
  max_results?: number;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  time_ms: number;
  latency_ms?: number | null;
  // Backwards-compatible fields from backend
  // (Some endpoints return latency_ms as number)
}

export interface Source {
  id: string;
  content: string;
  similarity: number;
  namespace: string;
}

export interface IngestRequest {
  content: string;
  filename?: string;
  source?: string;
  namespace: string;
  tags?: string[];
}

export interface IngestResponse {
  node_count: number;
  chunks: number;
  time_ms: number;
}

export interface RememberRequest {
  content: string;
  user_id: string;
  metadata?: Record<string, string>;
}

export interface RememberResponse {
  node_id: string;
  success: boolean;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  services: ServiceHealth[];
}

export interface ServiceHealth {
  name: string;
  status: 'up' | 'down';
  latency_ms?: number;
}

export interface RemPhaseStatus {
  running: boolean;
  last_run?: number;
  nodes_processed: number;
  clusters_formed: number;
}

// Node types for visualization
export type NodeType = 'Leaf' | 'Parent' | 'Root';
export type NodeStatus = 'Complete' | 'Incomplete' | 'Pending' | 'Deprecated';

export interface FractalNode {
  id: string;
  content: string;
  node_type: NodeType;
  status: NodeStatus;
  namespace: string;
  access_count: number;
  created_at: number;
  updated_at: number;
  tags: string[];
}

// Chat types
export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  created_at: number;
  updated_at: number;
}

// Settings
export interface Settings {
  apiUrl: string;
  namespace: string;
  userId: string;
  theme: 'light' | 'dark' | 'system';
  contextLimit: number;
}

// Re-export model types
export * from './models';
