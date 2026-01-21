import { useState, useCallback } from 'react';
import { api } from '@/services/api';
import type { HealthResponse, IngestResponse, RemPhaseStatus } from '@/types';

interface UseApiState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
}

export function useHealth() {
  const [state, setState] = useState<UseApiState<HealthResponse>>({
    data: null,
    isLoading: false,
    error: null,
  });

  const checkHealth = useCallback(async () => {
    setState({ data: null, isLoading: true, error: null });
    try {
      const data = await api.health();
      setState({ data, isLoading: false, error: null });
      return data;
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error';
      setState({ data: null, isLoading: false, error });
      throw err;
    }
  }, []);

  return { ...state, checkHealth };
}

export function useIngest() {
  const [state, setState] = useState<UseApiState<IngestResponse>>({
    data: null,
    isLoading: false,
    error: null,
  });

  const ingestFile = useCallback(
    async (file: File, namespace: string, tags?: string[]) => {
      setState({ data: null, isLoading: true, error: null });
      try {
        const data = await api.ingestFile(file, namespace, tags);
        setState({ data, isLoading: false, error: null });
        return data;
      } catch (err) {
        const error = err instanceof Error ? err.message : 'Unknown error';
        setState({ data: null, isLoading: false, error });
        throw err;
      }
    },
    []
  );

  const ingestText = useCallback(
    async (content: string, namespace: string, source?: string, tags?: string[]) => {
      setState({ data: null, isLoading: true, error: null });
      try {
        const data = await api.ingest({ content, namespace, source, tags });
        setState({ data, isLoading: false, error: null });
        return data;
      } catch (err) {
        const error = err instanceof Error ? err.message : 'Unknown error';
        setState({ data: null, isLoading: false, error });
        throw err;
      }
    },
    []
  );

  return { ...state, ingestFile, ingestText };
}

export function useRemPhase() {
  const [state, setState] = useState<UseApiState<RemPhaseStatus>>({
    data: null,
    isLoading: false,
    error: null,
  });

  const triggerRem = useCallback(async () => {
    setState({ data: null, isLoading: true, error: null });
    try {
      const data = await api.triggerRem();
      setState({ data, isLoading: false, error: null });
      return data;
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error';
      setState({ data: null, isLoading: false, error });
      throw err;
    }
  }, []);

  const getStatus = useCallback(async () => {
    setState({ data: null, isLoading: true, error: null });
    try {
      const data = await api.remStatus();
      setState({ data, isLoading: false, error: null });
      return data;
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error';
      setState({ data: null, isLoading: false, error });
      throw err;
    }
  }, []);

  return { ...state, triggerRem, getStatus };
}
