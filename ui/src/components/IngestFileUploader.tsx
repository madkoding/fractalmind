import React, { useRef, useState } from 'react';
import { api } from '@/services/api';
import { useIngest } from '@/hooks/useApi';

export function IngestFileUploader({
  defaultNamespace = 'global_knowledge',
  onSuccess,
  onError,
}: {
  defaultNamespace?: string;
  onSuccess?: (res: any) => void;
  onError?: (err: Error) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [namespace, setNamespace] = useState(defaultNamespace);
  const [tags, setTags] = useState('');
  const [isUploading, setUploading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const { ingestFile } = useIngest();

  const onPickFile = () => fileInputRef.current?.click();

  const handleFile = async (file?: File | null) => {
    if (!file) return;
    setUploading(true);
    setMessage(null);

    try {
      const tagList = tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      const res = await ingestFile(file, namespace, tagList);
      setMessage('File uploaded successfully');
      onSuccess?.(res);
    } catch (err: any) {
      const msg = err?.message || 'Upload failed';
      setMessage(msg);
      onError?.(err instanceof Error ? err : new Error(msg));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-3 bg-gray-900 border-t border-gray-700">
      <div className="max-w-4xl mx-auto flex items-center gap-2">
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          data-testid="file-input"
        />

        <button
          type="button"
          className="px-3 py-2 rounded bg-gray-800 hover:bg-gray-700 text-white"
          onClick={onPickFile}
          disabled={isUploading}
          data-testid="pick-file"
        >
          Attach file
        </button>

        <input
          value={namespace}
          onChange={(e) => setNamespace(e.target.value)}
          className="flex-1 px-3 py-2 bg-gray-800 rounded text-white"
          placeholder="Namespace"
          data-testid="namespace-input"
        />

        <input
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          className="w-48 px-3 py-2 bg-gray-800 rounded text-white"
          placeholder="Tags (comma separated)"
          data-testid="tags-input"
        />

        <div className="w-40 text-right">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-2 rounded bg-fractal-500 text-white hover:opacity-90 disabled:opacity-50"
            disabled={isUploading}
            data-testid="upload-button"
          >
            {isUploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>

      {message && <p className="mt-2 text-sm text-gray-300" data-testid="message">{message}</p>}
    </div>
  );
}

export default IngestFileUploader;
