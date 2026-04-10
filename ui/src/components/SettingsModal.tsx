import { useState, useEffect } from 'react';
import { X, Check, RotateCcw, Server, Cloud, AlertCircle, CheckCircle, XCircle, Brain } from 'lucide-react';
import { useSettingsStore } from '@/stores/settingsStore';
import { api } from '@/services/api';
import type { LLMConfigStatus, UpdateLLMConfigRequest } from '@/types/models';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';
import { SUPPORTED_LANGUAGES } from '@/i18n';
import type { LanguageMode, SupportedLanguage } from '@/types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'general' | 'ai-model';

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { t } = useTranslation();
  const settings = useSettingsStore();
  const [activeTab, setActiveTab] = useState<TabType>('general');
  
  const [localSettings, setLocalSettings] = useState({
    namespace: settings.namespace,
    userId: settings.userId,
    theme: settings.theme,
    language: settings.language,
    languageMode: settings.languageMode,
    detectedLanguage: settings.detectedLanguage,
  });

  // LLM Provider state
  const [llmConfig, setLlmConfig] = useState<LLMConfigStatus | null>(null);
  const [providerType, setProviderType] = useState<'ollama' | 'ollama-cloud'>('ollama');
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState('http://localhost:11434');
  const [ollamaApiKey, setOllamaApiKey] = useState('');
  const [isLoadingConfig, setIsLoadingConfig] = useState(false);
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchLLMConfig();
    }
  }, [isOpen]);

  const fetchLLMConfig = async () => {
    setIsLoadingConfig(true);
    setConfigError(null);
    try {
      const config = await api.getLLMConfig();
      setLlmConfig(config);
      setProviderType(config.is_cloud ? 'ollama-cloud' : 'ollama');
      setOllamaBaseUrl(config.ollama_base_url);
    } catch (err) {
      console.error('Failed to fetch LLM config:', err);
      setConfigError(t('settings.error.fetchConfig'));
    } finally {
      setIsLoadingConfig(false);
    }
  };

  const handleSaveLLMConfig = async () => {
    setIsSavingConfig(true);
    setConfigError(null);
    try {
      const request: UpdateLLMConfigRequest = {
        provider_type: providerType,
        ollama_base_url: providerType === 'ollama' ? ollamaBaseUrl : undefined,
        ollama_api_key: providerType === 'ollama-cloud' && ollamaApiKey ? ollamaApiKey : undefined,
      };
      const response = await api.updateLLMConfig(request);
      setLlmConfig(response.config);
      setOllamaApiKey('');
    } catch (err) {
      console.error('Failed to save LLM config:', err);
      setConfigError(t('settings.error.saveConfig'));
    } finally {
      setIsSavingConfig(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      setLocalSettings({
        namespace: settings.namespace,
        userId: settings.userId,
        theme: settings.theme,
        language: settings.language,
        languageMode: settings.languageMode,
        detectedLanguage: settings.detectedLanguage,
      });
    }
  }, [isOpen, settings]);

  if (!isOpen) return null;

  const handleSave = async () => {
    settings.setNamespace(localSettings.namespace);
    settings.setUserId(localSettings.userId);
    settings.setTheme(localSettings.theme);
    if (localSettings.languageMode === 'auto') {
      await settings.setLanguageMode('auto');
    } else {
      await settings.setLanguage(localSettings.language);
    }
    onClose();
  };

  const handleReset = () => {
    settings.resetSettings();
    onClose();
  };

  const getStatusBadge = (healthy: boolean) => {
    if (healthy) {
      return <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 text-xs"><CheckCircle className="w-3 h-3" /> {t('settings.status.active')}</span>;
    }
    return <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 text-xs"><XCircle className="w-3 h-3" /> {t('settings.status.unavailable')}</span>;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-lg bg-gray-800 rounded-xl shadow-xl border border-gray-700 animate-in">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white">{t('settings.title')}</h2>
          <button onClick={onClose} className="p-1 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => setActiveTab('general')}
            className={clsx(
              'flex-1 px-4 py-3 text-sm font-medium transition-colors',
              activeTab === 'general'
                ? 'text-fractal-400 border-b-2 border-fractal-500'
                : 'text-gray-400 hover:text-white'
            )}
          >
            {t('settings.tabs.general')}
          </button>
          <button
            onClick={() => setActiveTab('ai-model')}
            className={clsx(
              'flex-1 px-4 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2',
              activeTab === 'ai-model'
                ? 'text-fractal-400 border-b-2 border-fractal-500'
                : 'text-gray-400 hover:text-white'
            )}
          >
            <Brain className="w-4 h-4" />
            {t('settings.tabs.aiModel')}
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {activeTab === 'general' && (
            <div className="space-y-4">
              {/* Namespace */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  {t('settings.memorySpace')}
                </label>
                <select
                  value={localSettings.namespace}
                  onChange={(e) => setLocalSettings({ ...localSettings, namespace: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:border-fractal-500 focus:outline-none"
                >
                  <option value="global">{t('settings.namespace.global')}</option>
                  <option value="personal">{t('settings.namespace.personal')}</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {localSettings.namespace === 'global' ? t('settings.namespace.globalHelp') : t('settings.namespace.personalHelp')}
                </p>
              </div>

              {/* User ID */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  {t('settings.userName')}
                </label>
                <input
                  type="text"
                  value={localSettings.userId}
                  onChange={(e) => setLocalSettings({ ...localSettings, userId: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-fractal-500 focus:outline-none"
                  placeholder={t('settings.userNamePlaceholder')}
                />
              </div>

              {/* Theme */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  {t('settings.appearance')}
                </label>
                <div className="flex gap-2">
                  {(['dark', 'light', 'system'] as const).map((theme) => (
                    <button
                      key={theme}
                      onClick={() => setLocalSettings({ ...localSettings, theme })}
                      className={clsx(
                        'flex-1 px-3 py-2 rounded-lg border text-sm capitalize transition-colors',
                        localSettings.theme === theme
                          ? 'border-fractal-500 bg-fractal-500/10 text-fractal-400'
                          : 'border-gray-700 text-gray-400 hover:border-gray-600'
                      )}
                    >
                      {theme === 'dark' ? t('settings.theme.dark') : theme === 'light' ? t('settings.theme.light') : t('settings.theme.system')}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  {t('settings.language')}
                </label>
                <div className="flex gap-2 mb-2">
                  {(['auto', 'manual'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setLocalSettings({ ...localSettings, languageMode: mode as LanguageMode })}
                      className={clsx(
                        'flex-1 px-3 py-2 rounded-lg border text-sm transition-colors',
                        localSettings.languageMode === mode
                          ? 'border-fractal-500 bg-fractal-500/10 text-fractal-400'
                          : 'border-gray-700 text-gray-400 hover:border-gray-600'
                      )}
                    >
                      {mode === 'auto' ? t('settings.languageAuto') : t('settings.languageManual')}
                    </button>
                  ))}
                </div>
                {localSettings.languageMode === 'manual' ? (
                  <select
                    value={localSettings.language}
                    onChange={(e) => setLocalSettings({
                      ...localSettings,
                      language: e.target.value as SupportedLanguage,
                    })}
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:border-fractal-500 focus:outline-none"
                  >
                    {SUPPORTED_LANGUAGES.map((lang) => (
                      <option key={lang.code} value={lang.code}>{lang.label}</option>
                    ))}
                  </select>
                ) : (
                  <p className="mt-1 text-xs text-gray-500">
                    {t('settings.languageAutoHelp')}
                    <br />
                    {t('settings.languageDetected', {
                      language: SUPPORTED_LANGUAGES.find((lang) => lang.code === settings.detectedLanguage)?.label || settings.detectedLanguage,
                    })}
                  </p>
                )}
              </div>
            </div>
          )}

          {activeTab === 'ai-model' && (
            <div className="space-y-4">
              {isLoadingConfig ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin w-8 h-8 border-2 border-fractal-500 border-t-transparent rounded-full" />
                </div>
              ) : configError ? (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-400">
                  <AlertCircle className="w-5 h-5" />
                  {configError}
                </div>
              ) : (
                <>
                  {/* Provider Selection */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      {t('settings.model.location')}
                    </label>
                    <div className="grid grid-cols-2 gap-3">
                      <button
                        onClick={() => setProviderType('ollama')}
                        className={clsx(
                          'p-4 rounded-lg border-2 transition-all text-left',
                          providerType === 'ollama'
                            ? 'border-fractal-500 bg-fractal-500/10'
                            : 'border-gray-700 hover:border-gray-600'
                        )}
                      >
                        <Server className={clsx('w-6 h-6 mb-2', providerType === 'ollama' ? 'text-fractal-400' : 'text-gray-400')} />
                        <div className="font-medium text-white">{t('settings.model.local')}</div>
                        <div className="text-xs text-gray-400 mt-1">{t('settings.model.localHint')}</div>
                      </button>
                      <button
                        onClick={() => setProviderType('ollama-cloud')}
                        className={clsx(
                          'p-4 rounded-lg border-2 transition-all text-left',
                          providerType === 'ollama-cloud'
                            ? 'border-fractal-500 bg-fractal-500/10'
                            : 'border-gray-700 hover:border-gray-600'
                        )}
                      >
                        <Cloud className={clsx('w-6 h-6 mb-2', providerType === 'ollama-cloud' ? 'text-fractal-400' : 'text-gray-400')} />
                        <div className="font-medium text-white">{t('settings.model.cloud')}</div>
                        <div className="text-xs text-gray-400 mt-1">{t('settings.model.cloudHint')}</div>
                      </button>
                    </div>
                  </div>

                  {/* Local URL */}
                  {providerType === 'ollama' && (
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                          {t('settings.model.ollamaUrl')}
                        </label>
                      <input
                        type="url"
                        value={ollamaBaseUrl}
                        onChange={(e) => setOllamaBaseUrl(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:border-fractal-500 focus:outline-none"
                        placeholder="http://localhost:11434"
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        {t('settings.model.ollamaUrlHint')}
                      </p>
                    </div>
                  )}

                  {/* Cloud API Key */}
                  {providerType === 'ollama-cloud' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        {t('settings.model.apiKey')}
                      </label>
                      <input
                        type="password"
                        value={ollamaApiKey}
                        onChange={(e) => setOllamaApiKey(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:border-fractal-500 focus:outline-none"
                        placeholder="sk-..."
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        {t('settings.model.apiKeyHint')} <a href="https://ollama.com/cloud" target="_blank" rel="noopener" className="text-fractal-400 hover:underline">ollama.com/cloud</a>
                      </p>
                    </div>
                  )}

                  {/* Current Status */}
                  {llmConfig && (
                    <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-700">
                      <div className="text-sm font-medium text-gray-300 mb-2">{t('settings.model.currentStatus')}</div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-400">{t('settings.model.chat')}</span>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-gray-500">{llmConfig.chat_model}</span>
                            {getStatusBadge(llmConfig.health_status.chat)}
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-400">{t('settings.model.embedding')}</span>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-gray-500">{llmConfig.embedding_model}</span>
                            {getStatusBadge(llmConfig.health_status.embedding)}
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-400">{t('settings.model.provider')}</span>
                          <span className="text-sm text-white">
                            {llmConfig.is_cloud ? t('settings.model.providerCloud') : t('settings.model.providerLocal')}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Save Button */}
                  <button
                    onClick={handleSaveLLMConfig}
                    disabled={isSavingConfig}
                    className={clsx(
                      'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors',
                      isSavingConfig
                        ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                        : 'bg-fractal-600 hover:bg-fractal-700 text-white'
                    )}
                  >
                    {isSavingConfig ? (
                      <>
                        <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
                        {t('settings.model.saving')}
                      </>
                    ) : (
                      <>
                        <Check className="w-5 h-5" />
                        {t('settings.model.save')}
                      </>
                    )}
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-gray-700">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            {t('settings.reset')}
          </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-4 py-2 bg-fractal-600 hover:bg-fractal-700 text-white rounded-lg transition-colors"
          >
            <Check className="w-4 h-4" />
            {t('settings.done')}
          </button>
        </div>
      </div>
    </div>
  );
}
