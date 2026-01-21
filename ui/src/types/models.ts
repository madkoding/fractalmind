// Model Management Types

export interface FractalModel {
  id: string;
  name: string;
  status: 'uploading' | 'converting' | 'ready' | 'failed';
  architecture?: ModelArchitecture;
  file_size: number;
  created_at: string;
  upload_progress?: number;
  conversion_progress?: number;
  current_phase?: string;
}

export interface ModelArchitecture {
  model_type: string;
  n_layers: number;
  embedding_dim: number;
  vocab_size: number;
  n_heads: number;
  ffn_dim: number;
}

export interface UploadModelResponse {
  success: boolean;
  model_id: string;
  message: string;
}

export interface ConvertModelRequest {
  model_id: string;
}

export interface ConvertModelResponse {
  success: boolean;
  message: string;
}

export interface ListModelsResponse {
  models: FractalModel[];
}

export interface GetModelResponse {
  model: FractalModel;
}

export interface DeleteModelResponse {
  success: boolean;
  message: string;
}

export interface UpdateStrategyRequest {
  strategy: 'fractal' | 'ollama';
  model_id?: string;
}

export interface UpdateStrategyResponse {
  success: boolean;
  current_strategy: string;
  message: string;
}

export type ModelStrategy = 'fractal' | 'ollama';

// Ollama Model Types
export interface OllamaModelDetails {
  parent_model?: string;
  format?: string;
  family?: string;
  families?: string[];
  parameter_size?: string;
  quantization_level?: string;
}

export interface OllamaModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details?: OllamaModelDetails;
}

export interface ListOllamaModelsResponse {
  models: OllamaModel[];
}

// Chunked Upload Types
export interface InitUploadRequest {
  filename: string;
  total_size: number;
  chunk_size: number;
}

export interface InitUploadResponse {
  upload_id: string;
  chunk_size: number;
  total_chunks: number;
}

export interface UploadChunkResponse {
  success: boolean;
  chunk_index: number;
  chunks_received: number;
  total_chunks: number;
}

export interface ProgressResponse {
  upload_progress: number;
  conversion_progress: number;
  status: string;
  upload_speed_mbps?: number;
  chunks_received?: number;
  total_chunks?: number;
  current_phase?: string;
}

export interface FinalizeUploadResponse {
  success: boolean;
  model_id: string;
  message: string;
}

export interface CancelUploadResponse {
  success: boolean;
  message: string;
}
