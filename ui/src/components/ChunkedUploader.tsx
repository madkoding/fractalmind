import { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, X, CheckCircle, AlertCircle } from 'lucide-react';
import { api } from '@/services';
import type { ProgressResponse } from '@/types/models';

interface ChunkedUploaderProps {
  onUploadComplete: (modelId: string) => void;
  onCancel?: () => void;
}

const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB chunks
const MAX_CONCURRENT_CHUNKS = 3;

export const ChunkedUploader = ({ onUploadComplete, onCancel }: ChunkedUploaderProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll for progress updates
  useEffect(() => {
    if (!uploadId) return;

    const pollProgress = async () => {
      try {
        const progressData = await api.getProgress(uploadId);
        setProgress(progressData);

        // If conversion is complete, notify parent
        if (progressData.status === 'ready' && progressData.conversion_progress >= 100) {
          stopProgressPolling();
          onUploadComplete(uploadId);
        }

        // If failed, stop polling
        if (progressData.status === 'failed') {
          stopProgressPolling();
          setError('Upload or conversion failed');
        }
      } catch (err) {
        console.error('Failed to fetch progress:', err);
      }
    };

    // Start polling every 2 seconds
    progressIntervalRef.current = setInterval(pollProgress, 2000);
    pollProgress(); // Initial fetch

    return () => stopProgressPolling();
  }, [uploadId, onUploadComplete]);

  const stopProgressPolling = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    // Validate file
    if (!selectedFile.name.toLowerCase().endsWith('.gguf')) {
      setError('Only .gguf files are supported');
      return;
    }

    setFile(selectedFile);
    setError(null);
  };

  const startUpload = async () => {
    if (!file) return;

    try {
      setUploading(true);
      setError(null);
      abortControllerRef.current = new AbortController();

      // Initialize upload
      const initResponse = await api.initUpload({
        filename: file.name,
        total_size: file.size,
        chunk_size: CHUNK_SIZE,
      });

      setUploadId(initResponse.upload_id);
      const totalChunks = initResponse.total_chunks;

      // Upload chunks with concurrency control
      const chunks: Promise<void>[] = [];
      let activeUploads = 0;
      let completedChunks = 0;

      for (let i = 0; i < totalChunks; i++) {
        // Wait if we've reached max concurrency
        while (activeUploads >= MAX_CONCURRENT_CHUNKS) {
          await Promise.race(chunks);
        }

        // Check if aborted
        if (abortControllerRef.current?.signal.aborted) {
          throw new Error('Upload cancelled');
        }

        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunkBlob = file.slice(start, end);

        activeUploads++;
        const chunkPromise = (async () => {
          try {
            await api.uploadChunk(initResponse.upload_id, i, totalChunks, chunkBlob);
            completedChunks++;
          } catch (err) {
            throw new Error(`Failed to upload chunk ${i}: ${err}`);
          } finally {
            activeUploads--;
          }
        })();

        chunks.push(chunkPromise);
      }

      // Wait for all chunks to complete
      await Promise.all(chunks);

      // Finalize upload
      await api.finalizeUpload(initResponse.upload_id);

      // Progress polling will detect completion and notify parent
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
      
      // Cancel upload on server
      if (uploadId) {
        try {
          await api.cancelUpload(uploadId);
        } catch {
          // Ignore cancel errors
        }
      }
    }
  };

  const handleCancel = async () => {
    abortControllerRef.current?.abort();
    stopProgressPolling();

    if (uploadId) {
      try {
        await api.cancelUpload(uploadId);
      } catch (err) {
        console.error('Failed to cancel upload:', err);
      }
    }

    setFile(null);
    setUploadId(null);
    setUploading(false);
    setProgress(null);
    setError(null);

    onCancel?.();
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  return (
    <div className="space-y-4">
      {/* File Selection */}
      {!file && !uploading && (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
              : 'border-gray-300 dark:border-gray-700 hover:border-purple-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Drag and drop your GGUF file here, or
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="mt-2 text-sm font-medium text-purple-600 hover:text-purple-500"
          >
            select a file
          </button>
          <p className="mt-1 text-xs text-gray-500">
            Supports files up to 500GB
          </p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".gguf"
            onChange={handleFileInput}
          />
        </div>
      )}

      {/* File Selected - Ready to Upload */}
      {file && !uploading && (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex-1">
              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                {file.name}
              </h4>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {formatBytes(file.size)}
              </p>
            </div>
            <button
              onClick={() => setFile(null)}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          <button
            onClick={startUpload}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-lg font-medium transition-colors"
          >
            Start Upload
          </button>
        </div>
      )}

      {/* Upload Progress */}
      {uploading && progress && (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-gray-900 dark:text-white">
              {file?.name || 'Uploading...'}
            </h4>
            <button
              onClick={handleCancel}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Upload Progress Bar */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-blue-600 dark:text-blue-400">
                Upload: {progress.upload_progress.toFixed(1)}%
              </span>
              {progress.upload_speed_mbps && (
                <span className="text-xs text-gray-500">
                  {progress.upload_speed_mbps.toFixed(1)} MB/s
                </span>
              )}
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress.upload_progress}%` }}
              />
            </div>
            {progress.chunks_received !== undefined && progress.total_chunks !== undefined && (
              <p className="text-xs text-gray-500 mt-1">
                {progress.chunks_received} / {progress.total_chunks} chunks
              </p>
            )}
          </div>

          {/* Conversion Progress Bar */}
          {progress.conversion_progress > 0 && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-green-600 dark:text-green-400">
                  Conversion: {progress.conversion_progress.toFixed(1)}%
                </span>
                {progress.current_phase && (
                  <span className="text-xs text-gray-500">
                    {progress.current_phase}
                  </span>
                )}
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress.conversion_progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Status */}
          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
            {progress.status === 'uploading' && <>Uploading chunks...</>}
            {progress.status === 'finalizing' && <>Finalizing upload...</>}
            {progress.status === 'converting' && <>Converting to fractal structure...</>}
            {progress.status === 'ready' && (
              <>
                <CheckCircle className="h-4 w-4 text-green-600" />
                Complete!
              </>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-red-800 dark:text-red-200">Upload Error</p>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-600 dark:hover:text-red-300"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  );
};
