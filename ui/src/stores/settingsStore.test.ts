import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockSetI18nLanguage, mockDetectPreferredLanguage } = vi.hoisted(() => ({
  mockSetI18nLanguage: vi.fn(async () => undefined),
  mockDetectPreferredLanguage: vi.fn(async () => 'en'),
}));

vi.mock('@/i18n', () => ({
  DEFAULT_LANGUAGE: 'en',
  detectPreferredLanguage: mockDetectPreferredLanguage,
  normalizeLanguage: (value: string) => {
    if (value === 'zh' || value === 'zh-CN') return 'zh-CN';
    const normalized = value.split('-')[0];
    if (['en', 'es', 'pt', 'fr', 'de', 'it', 'ru', 'ja'].includes(normalized)) {
      return normalized;
    }
    return 'en';
  },
  setI18nLanguage: mockSetI18nLanguage,
}));

import { useSettingsStore } from './settingsStore';

describe('settingsStore language', () => {
  beforeEach(() => {
    localStorage.clear();
    mockSetI18nLanguage.mockClear();
    mockDetectPreferredLanguage.mockReset();
    mockDetectPreferredLanguage.mockResolvedValue('en');

    useSettingsStore.setState({
      apiUrl: 'http://localhost:9000',
      namespace: 'global_knowledge',
      userId: '',
      theme: 'dark',
      contextLimit: 10,
      language: 'en',
      languageMode: 'auto',
      detectedLanguage: 'en',
    });
  });

  it('sets manual language and switches mode to manual', async () => {
    await useSettingsStore.getState().setLanguage('es');

    const state = useSettingsStore.getState();
    expect(state.language).toBe('es');
    expect(state.languageMode).toBe('manual');
    expect(state.detectedLanguage).toBe('es');
    expect(mockSetI18nLanguage).toHaveBeenCalledWith('es');
  });

  it('uses auto-detected language in auto mode', async () => {
    mockDetectPreferredLanguage.mockResolvedValue('ja');

    await useSettingsStore.getState().setLanguageMode('auto');

    const state = useSettingsStore.getState();
    expect(state.languageMode).toBe('auto');
    expect(state.language).toBe('ja');
    expect(state.detectedLanguage).toBe('ja');
    expect(mockSetI18nLanguage).toHaveBeenCalledWith('ja');
  });
});
