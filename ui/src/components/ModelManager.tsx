import { useState, useEffect, useCallback } from 'react';
import { 
  Upload, 
  Trash2, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  Loader2,
  Database,
  Zap,
  Plus
} from 'lucide-react';
import { api } from '@/services';
import { ChunkedUploader } from './ChunkedUploader';
import type { FractalModel, ModelStrategy, OllamaModel } from '@/types';

export const ModelManager = () => {
  const [models, setModels] = useState<FractalModel[]>([]);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUploader, setShowUploader] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<ModelStrategy>('ollama');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Load models on mount and when strategy changes
  useEffect(() => {
    loadModels();
  }, [strategy]);

  const loadModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      if (strategy === 'ollama') {
        const response = await api.listOllamaModels();
        setOllamaModels(response.models);
      } else {
        const response = await api.listModels();
        setModels(response.models);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoading(false);
    }
  }, [strategy]);

  const handleConvert = async (modelId: string) => {
    try {
      setError(null);
      await api.convertModel({ model_id: modelId });
      
      // Update model status optimistically
      setModels(prev =>
        prev.map(m =>
          m.id === modelId ? { ...m, status: 'converting' } : m
        )
      );
      
      // Refresh after delay
      setTimeout(loadModels, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Conversion failed');
    }
  };

  const handleDelete = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model?')) {
      return;
    }

    try {
      setError(null);
      await api.deleteModel(modelId);
      setModels(prev => prev.filter(m => m.id !== modelId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    }
  };

  const handleStrategyChange = async (newStrategy: ModelStrategy) => {
    try {
      setError(null);
      await api.updateStrategy({
        strategy: newStrategy,
        model_id: newStrategy === 'fractal' ? selectedModel ?? undefined : undefined,
      });
      setStrategy(newStrategy);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Strategy update failed');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  const getStatusIcon = (status: FractalModel['status']) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'converting':
      case 'uploading':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Loader2 className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: FractalModel['status']) => {
    switch (status) {
      case 'ready':
        return 'bg-green-500/10 text-green-400 border-green-500/30';
      case 'failed':
        return 'bg-red-500/10 text-red-400 border-red-500/30';
      case 'converting':
      case 'uploading':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/30';
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-800 p-6">
        <h1 className="text-2xl font-bold mb-2">Fractal Model Manager</h1>
        <p className="text-gray-400">Upload and manage GGUF models for fractal conversion</p>
      </div>

      {/* Strategy Selector */}
      <div className="border-b border-gray-800 p-4">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-gray-400">Inference Strategy:</span>
          <div className="flex gap-2">
            <button
              onClick={() => handleStrategyChange('ollama')}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                strategy === 'ollama'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              <Zap className="w-4 h-4" />
              Ollama (Direct)
            </button>
            <button
              onClick={() => {
                setStrategy('fractal');
                loadModels();
              }}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                strategy === 'fractal'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              <Database className="w-4 h-4" />
              Fractal (Graph)
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="m-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
          {error}
        </div>
      )}

      {/* Upload Area */}
      <div className="p-6">
        {showUploader ? (
          <ChunkedUploader
            onUploadComplete={(modelId) => {
              setShowUploader(false);
              loadModels();
            }}
            onCancel={() => setShowUploader(false)}
          />
        ) : (
          <button
            onClick={() => setShowUploader(true)}
            className="w-full border-2 border-dashed border-gray-700 hover:border-blue-500 rounded-lg p-8 text-center transition-colors group"
          >
            <Plus className="w-12 h-12 mx-auto mb-4 text-gray-400 group-hover:text-blue-400" />
            <p className="text-lg mb-2 text-gray-300">Upload GGUF Model</p>
            <p className="text-sm text-gray-500">
              Click to upload a model file (supports files up to 1TB with chunked upload)
            </p>
          </button>
        )}
      </div>

      {/* Models List */}
      <div className="flex-1 overflow-auto p-6 pt-0">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">
            {strategy === 'ollama' 
              ? `Ollama Models (${ollamaModels.length})` 
              : `Fractal Models (${models.length})`}
          </h2>
          <button
            onClick={loadModels}
            disabled={loading}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {loading && (strategy === 'ollama' ? ollamaModels.length === 0 : models.length === 0) ? (
          <div className="text-center py-12 text-gray-500">
            <Loader2 className="w-8 h-8 mx-auto mb-2 animate-spin" />
            Loading models...
          </div>
        ) : strategy === 'ollama' ? (
          // Ollama Models List
          ollamaModels.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              No Ollama models found. Make sure Ollama is running with models installed.
            </div>
          ) : (
            <div className="grid gap-4">
              {ollamaModels.map(model => (
                <div
                  key={model.digest}
                  className={`bg-gray-800 rounded-lg p-4 border ${
                    selectedModel === model.name
                      ? 'border-blue-500'
                      : 'border-gray-700 hover:border-gray-600'
                  } transition-colors cursor-pointer`}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <CheckCircle className="w-5 h-5 text-green-500" />
                        <h3 className="font-semibold text-lg">{model.name}</h3>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm text-gray-400">
                        <div>
                          <span className="font-medium">Size:</span> {formatFileSize(model.size)}
                        </div>
                        <div>
                          <span className="font-medium">Modified:</span>{' '}
                          {new Date(model.modified_at).toLocaleDateString()}
                        </div>
                        
                        {model.details && (
                          <>
                            {model.details.parameter_size && (
                              <div>
                                <span className="font-medium">Parameters:</span> {model.details.parameter_size}
                              </div>
                            )}
                            {model.details.quantization_level && (
                              <div>
                                <span className="font-medium">Quantization:</span> {model.details.quantization_level}
                              </div>
                            )}
                            {model.details.family && (
                              <div>
                                <span className="font-medium">Family:</span> {model.details.family}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )
        ) : (
          // Fractal Models List
          models.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              No models uploaded yet. Upload a GGUF model to get started.
            </div>
          ) : (
            <div className="grid gap-4">
              {models.map(model => (
                <div
                  key={model.id}
                  className={`bg-gray-800 rounded-lg p-4 border ${
                    selectedModel === model.id
                      ? 'border-purple-500'
                      : 'border-gray-700 hover:border-gray-600'
                  } transition-colors cursor-pointer`}
                  onClick={() => setSelectedModel(model.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        {getStatusIcon(model.status)}
                        <h3 className="font-semibold text-lg">{model.name}</h3>
                        <span
                          className={`px-2 py-1 rounded text-xs border ${getStatusColor(
                            model.status
                          )}`}
                        >
                          {model.status}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm text-gray-400">
                        <div>
                          <span className="font-medium">Size:</span> {formatFileSize(model.file_size)}
                        </div>
                        <div>
                          <span className="font-medium">Created:</span>{' '}
                          {new Date(model.created_at).toLocaleDateString()}
                        </div>
                        
                        {model.architecture && (
                          <>
                            <div>
                              <span className="font-medium">Type:</span> {model.architecture.model_type}
                            </div>
                            <div>
                              <span className="font-medium">Layers:</span> {model.architecture.n_layers}
                            </div>
                            <div>
                              <span className="font-medium">Embedding:</span> {model.architecture.embedding_dim}D
                            </div>
                            <div>
                              <span className="font-medium">Vocab:</span>{' '}
                              {model.architecture.vocab_size.toLocaleString()}
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="flex gap-2">
                      {model.status === 'ready' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleConvert(model.id);
                          }}
                          className="p-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                          title="Convert to fractal"
                        >
                          <RefreshCw className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(model.id);
                        }}
                        className="p-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg transition-colors"
                        title="Delete model"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  );
};
