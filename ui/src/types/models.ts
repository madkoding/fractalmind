// Model Management Types

export interface FractalModel {
  id: string;
  name: string;
  status: 'uploading' | 'converting' | 'ready' | 'failed';
  architecture?: ModelArchitecture;
  file_size: number;
  created_at: string;
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
