import { useChatStore } from '@/stores/chatStore';
import { MessageSquarePlus, Trash2, Brain, Settings } from 'lucide-react';
import clsx from 'clsx';

interface SidebarProps {
  onSettingsClick: () => void;
}

export function Sidebar({ onSettingsClick }: SidebarProps) {
  const {
    conversations,
    currentConversationId,
    createConversation,
    selectConversation,
    deleteConversation,
  } = useChatStore();

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
          New Chat
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 ? (
          <p className="text-gray-500 text-sm text-center py-4">
            No conversations yet
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
                    title="Delete conversation"
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
        <button
          onClick={onSettingsClick}
          className="w-full flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
      </div>
    </div>
  );
}
