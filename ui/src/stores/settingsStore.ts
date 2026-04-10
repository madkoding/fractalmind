import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Settings } from '@/types';
import { api } from '@/services/api';
import type { LanguageMode, SupportedLanguage } from '@/types';
import {
  DEFAULT_LANGUAGE,
  detectPreferredLanguage,
  normalizeLanguage,
  setI18nLanguage,
} from '@/i18n';

interface SettingsState extends Settings {
  setApiUrl: (url: string) => void;
  setNamespace: (namespace: string) => void;
  setUserId: (userId: string) => void;
  setTheme: (theme: Settings['theme']) => void;
  setContextLimit: (limit: number) => void;
  setLanguageMode: (mode: LanguageMode) => Promise<void>;
  setLanguage: (language: SupportedLanguage) => Promise<void>;
  detectedLanguage: SupportedLanguage;
  initializeLanguage: () => Promise<void>;
  resetSettings: () => void;
}

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:9000';

const DEFAULT_SETTINGS: Settings = {
  apiUrl: DEFAULT_API_URL,
  namespace: 'global_knowledge',
  userId: '',
  theme: 'dark',
  contextLimit: 10,
  language: DEFAULT_LANGUAGE,
  languageMode: 'auto',
};

async function applyLanguage(mode: LanguageMode, language: SupportedLanguage): Promise<SupportedLanguage> {
  const nextLanguage = mode === 'auto' ? await detectPreferredLanguage() : normalizeLanguage(language);
  await setI18nLanguage(nextLanguage);
  return nextLanguage;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,
      detectedLanguage: DEFAULT_LANGUAGE,

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

      setLanguageMode: async (mode: LanguageMode) => {
        const current = useSettingsStore.getState();
        const resolvedLanguage = await applyLanguage(mode, current.language);
        set({
          languageMode: mode,
          language: resolvedLanguage,
          detectedLanguage: resolvedLanguage,
        });
      },

      setLanguage: async (language: SupportedLanguage) => {
        const normalized = normalizeLanguage(language);
        await setI18nLanguage(normalized);
        set({
          language: normalized,
          languageMode: 'manual',
          detectedLanguage: normalized,
        });
      },

      initializeLanguage: async () => {
        const current = useSettingsStore.getState();
        const resolvedLanguage = await applyLanguage(current.languageMode, current.language);
        set({
          language: resolvedLanguage,
          detectedLanguage: resolvedLanguage,
        });
      },

      resetSettings: () => {
        api.setBaseUrl(DEFAULT_SETTINGS.apiUrl);
        set({
          ...DEFAULT_SETTINGS,
          detectedLanguage: DEFAULT_LANGUAGE,
        });
        void applyLanguage(DEFAULT_SETTINGS.languageMode, DEFAULT_SETTINGS.language).then((resolvedLanguage) => {
          useSettingsStore.setState({
            language: resolvedLanguage,
            languageMode: DEFAULT_SETTINGS.languageMode,
            detectedLanguage: resolvedLanguage,
          });
        });
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

        if (state) {
          const languageMode = state.languageMode || 'auto';
          const language = normalizeLanguage(state.language || DEFAULT_LANGUAGE);
          void applyLanguage(languageMode, language).then((resolvedLanguage) => {
            useSettingsStore.setState({
              language: resolvedLanguage,
              languageMode,
              detectedLanguage: resolvedLanguage,
            });
          });
        }
      },
    }
  )
);
