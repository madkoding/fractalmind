import type { SupportedLanguage } from '@/types';

export const DEFAULT_LANGUAGE: SupportedLanguage = 'en';

export const SUPPORTED_LANGUAGES: Array<{ code: SupportedLanguage; label: string }> = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Espanol' },
  { code: 'pt', label: 'Portugues' },
  { code: 'fr', label: 'Francais' },
  { code: 'de', label: 'Deutsch' },
  { code: 'it', label: 'Italiano' },
  { code: 'ru', label: 'Russkiy' },
  { code: 'ja', label: 'Nihongo' },
  { code: 'zh-CN', label: 'Zhongwen (Jianti)' },
];

export const SUPPORTED_LANGUAGE_CODES = SUPPORTED_LANGUAGES.map((item) => item.code);

export function isSupportedLanguage(value: string): value is SupportedLanguage {
  return SUPPORTED_LANGUAGE_CODES.includes(value as SupportedLanguage);
}
