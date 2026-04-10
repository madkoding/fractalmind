import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import type { SupportedLanguage } from '@/types';
import { DEFAULT_LANGUAGE, isSupportedLanguage } from './languages';
import { resources } from './resources';

function toHtmlLang(value: SupportedLanguage): string {
  return value === 'zh-CN' ? 'zh-CN' : value;
}

export function applyDocumentLanguage(language: SupportedLanguage): void {
  document.documentElement.setAttribute('lang', toHtmlLang(language));
}

export function normalizeLanguage(value: string): SupportedLanguage {
  if (isSupportedLanguage(value)) {
    return value;
  }

  const base = value.split('-')[0]?.toLowerCase();
  if (!base) {
    return DEFAULT_LANGUAGE;
  }

  if (base === 'zh') {
    return 'zh-CN';
  }

  return isSupportedLanguage(base) ? base : DEFAULT_LANGUAGE;
}

export async function initI18n(initialLanguage: SupportedLanguage = DEFAULT_LANGUAGE): Promise<void> {
  if (!i18n.isInitialized) {
    await i18n.use(initReactI18next).init({
      resources,
      lng: initialLanguage,
      fallbackLng: DEFAULT_LANGUAGE,
      interpolation: {
        escapeValue: false,
      },
      returnNull: false,
    });
  }

  applyDocumentLanguage(normalizeLanguage(i18n.language));
}

export async function setI18nLanguage(language: SupportedLanguage): Promise<void> {
  const normalized = normalizeLanguage(language);
  await i18n.changeLanguage(normalized);
  applyDocumentLanguage(normalized);
}

export default i18n;
