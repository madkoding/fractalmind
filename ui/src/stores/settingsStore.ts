import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Settings } from '@/types';
import { api } from '@/services/api';

interface SettingsState extends Settings {
  setApiUrl: (url: string) => void;
  setNamespace: (namespace: string) => void;
  setUserId: (userId: string) => void;
  setTheme: (theme: Settings['theme']) => void;
  setContextLimit: (limit: number) => void;
  resetSettings: () => void;
}

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

const DEFAULT_SETTINGS: Settings = {
  apiUrl: DEFAULT_API_URL,
  namespace: 'global',
  userId: '',
  theme: 'dark',
  contextLimit: 10,
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,

      setApiUrl: (url: string) => {
        api.setBaseUrl(url);
        set({ apiUrl: url });
      },

      setNamespace: (namespace: string) => {
        set({ namespace });
      },

      setUserId: (userId: string) => {
        set({ userId });
      },

      setTheme: (theme: Settings['theme']) => {
        set({ theme });
        // Apply theme to document
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else if (theme === 'light') {
          document.documentElement.classList.remove('dark');
        } else {
          // System preference
          const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          document.documentElement.classList.toggle('dark', prefersDark);
        }
      },

      setContextLimit: (limit: number) => {
        set({ contextLimit: Math.max(1, Math.min(50, limit)) });
      },

      resetSettings: () => {
        api.setBaseUrl(DEFAULT_SETTINGS.apiUrl);
        set(DEFAULT_SETTINGS);
      },
    }),
    {
      name: 'fractalmind-settings',
      onRehydrateStorage: () => (state) => {
        // Apply stored API URL on load
        if (state?.apiUrl) {
          api.setBaseUrl(state.apiUrl);
        }
        // Apply theme
        if (state?.theme) {
          const theme = state.theme;
          if (theme === 'dark') {
            document.documentElement.classList.add('dark');
          } else if (theme === 'light') {
            document.documentElement.classList.remove('dark');
          }
        }
      },
    }
  )
);
