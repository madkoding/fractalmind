import { useEffect, useMemo, useState } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { MessageSquarePlus, Trash2, Brain, Settings } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';
import { useSettingsStore } from '@/stores/settingsStore';
import { SUPPORTED_LANGUAGES } from '@/i18n';
import { api } from '@/services/api';
import type { ServiceStatus, SystemStatus } from '@/types';

interface SidebarProps {
  onSettingsClick: () => void;
}

export function Sidebar({ onSettingsClick }: SidebarProps) {
  const { t } = useTranslation();
  const {
    language,
    languageMode,
    apiUrl,
    setLanguage,
    setLanguageMode,
  } = useSettingsStore();

  const {
    conversations,
    currentConversationId,
    createConversation,
    selectConversation,
    deleteConversation,
  } = useChatStore();

  const languageValue = languageMode === 'auto' ? '__auto__' : language;
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let shouldReconnect = true;

    const connect = () => {
      socket = new WebSocket(api.getStatusWebSocketUrl());

      socket.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as SystemStatus;
          if (parsed?.services) {
            setSystemStatus(parsed);
          }
        } catch {
          // Ignore malformed payloads
        }
      };

      socket.onclose = () => {
        if (shouldReconnect) {
          reconnectTimer = window.setTimeout(connect, 3000);
        }
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      shouldReconnect = false;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [apiUrl]);

  const services = useMemo(() => {
    const defaults: ServiceStatus[] = [
      { name: 'surrealdb', healthy: false, message: 'checking' },
      { name: 'ollama', healthy: false, message: 'checking' },
      { name: 'chat_provider', healthy: false, message: 'checking' },
      { name: 'searxng', healthy: false, message: 'checking' },
    ];

    if (!systemStatus?.services?.length) {
      return defaults;
    }
    return systemStatus.services;
  }, [systemStatus]);

  const handleLanguageChange = (value: string) => {
    if (value === '__auto__') {
      void setLanguageMode('auto');
      return;
    }

    void setLanguage(value as typeof language);
  };

  return (
    <div className="w-64 bg-gray-800 flex flex-col h-full border-r border-gray-700">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-8 h-8 text-fractal-500" />
          <span className="text-xl font-bold text-white">Fractal-Mind</span>
        </div>
        
        <button
          onClick={() => createConversation()}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-fractal-600 hover:bg-fractal-700 text-white rounded-lg transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4" />
          {t('sidebar.newChat')}
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 ? (
          <p className="text-gray-500 text-sm text-center py-4">
            {t('sidebar.noConversations')}
          </p>
        ) : (
          <ul className="space-y-1">
            {conversations.map((conversation) => (
              <li key={conversation.id}>
                <button
                  onClick={() => selectConversation(conversation.id)}
                  className={clsx(
                    'w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors group',
                    currentConversationId === conversation.id
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-400 hover:bg-gray-700/50 hover:text-white'
                  )}
                >
                  <span className="truncate flex-1 text-left">
                    {conversation.title}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conversation.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-600 rounded transition-all"
                    title={t('sidebar.deleteConversation')}
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <div className="mb-3 rounded-lg border border-gray-700 bg-gray-900/60 p-2">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            {t('sidebar.services')}
          </div>
          <div className="space-y-1.5">
            {services.map((service) => (
              <div key={service.name} className="flex items-center justify-between text-xs">
                <span className="text-gray-300">{service.name}</span>
                <span className="inline-flex items-center gap-1.5 text-gray-300">
                  <span
                    className={clsx(
                      'h-2 w-2 rounded-full',
                      service.healthy ? 'bg-green-400 shadow-[0_0_8px_#4ade80]' : 'bg-red-400 shadow-[0_0_8px_#f87171]'
                    )}
                  />
                  {service.healthy ? t('sidebar.serviceOk') : t('sidebar.serviceDown')}
                </span>
              </div>
            ))}
          </div>
        </div>

        <label className="block text-xs text-gray-500 mb-2">
          {t('sidebar.language')}
        </label>
        <select
          value={languageValue}
          onChange={(e) => handleLanguageChange(e.target.value)}
          className="w-full mb-3 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm text-white focus:border-fractal-500 focus:outline-none"
        >
          <option value="__auto__">{t('settings.languageAuto')}</option>
          {SUPPORTED_LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>{lang.label}</option>
          ))}
        </select>

        <button
          onClick={onSettingsClick}
          className="w-full flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
          {t('sidebar.settings')}
        </button>
      </div>
    </div>
  );
}
