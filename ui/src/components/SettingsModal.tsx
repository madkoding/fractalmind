import { useState, useEffect } from 'react';
import { X, Check, RotateCcw } from 'lucide-react';
import { useSettingsStore } from '@/stores/settingsStore';
import clsx from 'clsx';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const settings = useSettingsStore();
  const [localSettings, setLocalSettings] = useState({
    apiUrl: settings.apiUrl,
    namespace: settings.namespace,
    userId: settings.userId,
    contextLimit: settings.contextLimit,
    theme: settings.theme,
  });

  // Sync local state when modal opens
  useEffect(() => {
    if (isOpen) {
      setLocalSettings({
        apiUrl: settings.apiUrl,
        namespace: settings.namespace,
        userId: settings.userId,
        contextLimit: settings.contextLimit,
        theme: settings.theme,
      });
    }
  }, [isOpen, settings]);

  if (!isOpen) return null;

  const handleSave = () => {
    settings.setApiUrl(localSettings.apiUrl);
    settings.setNamespace(localSettings.namespace);
    settings.setUserId(localSettings.userId);
    settings.setContextLimit(localSettings.contextLimit);
    settings.setTheme(localSettings.theme);
    onClose();
  };

  const handleReset = () => {
    settings.resetSettings();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-md bg-gray-800 rounded-xl shadow-xl border border-gray-700 animate-in">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* API URL */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              API URL
            </label>
            <input
              type="url"
              value={localSettings.apiUrl}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, apiUrl: e.target.value })
              }
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-fractal-500 focus:outline-none"
              placeholder={import.meta.env.VITE_API_URL || "http://localhost:3000"}
            />
          </div>

          {/* Namespace */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Namespace
            </label>
            <select
              value={localSettings.namespace}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, namespace: e.target.value })
              }
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:border-fractal-500 focus:outline-none"
            >
              <option value="global">Global Knowledge</option>
              <option value="personal">Personal Memory</option>
            </select>
          </div>

          {/* User ID */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              User ID (optional)
            </label>
            <input
              type="text"
              value={localSettings.userId}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, userId: e.target.value })
              }
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-fractal-500 focus:outline-none"
              placeholder="your-user-id"
            />
          </div>

          {/* Context Limit */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Context Limit: {localSettings.contextLimit}
            </label>
            <input
              type="range"
              min="1"
              max="50"
              value={localSettings.contextLimit}
              onChange={(e) =>
                setLocalSettings({
                  ...localSettings,
                  contextLimit: parseInt(e.target.value, 10),
                })
              }
              className="w-full accent-fractal-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>50</span>
            </div>
          </div>

          {/* Theme */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Theme
            </label>
            <div className="flex gap-2">
              {(['dark', 'light', 'system'] as const).map((theme) => (
                <button
                  key={theme}
                  onClick={() =>
                    setLocalSettings({ ...localSettings, theme })
                  }
                  className={clsx(
                    'flex-1 px-3 py-2 rounded-lg border text-sm capitalize transition-colors',
                    localSettings.theme === theme
                      ? 'border-fractal-500 bg-fractal-500/10 text-fractal-400'
                      : 'border-gray-700 text-gray-400 hover:border-gray-600'
                  )}
                >
                  {theme}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-gray-700">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-4 py-2 bg-fractal-600 hover:bg-fractal-700 text-white rounded-lg transition-colors"
            >
              <Check className="w-4 h-4" />
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
