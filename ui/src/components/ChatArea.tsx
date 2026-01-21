import { useEffect, useRef } from 'react';
import { useChatStore, useMessages } from '@/stores/chatStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { ChatMessage, ChatMessageLoading } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { Brain, Sparkles } from 'lucide-react';

export function ChatArea() {
  const messages = useMessages();
  const { isLoading, error, sendMessage, clearError } = useChatStore();
  const { namespace } = useSettingsStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(clearError, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, clearError]);

  const handleSend = (content: string) => {
    sendMessage(content, namespace);
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-900 h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="max-w-4xl mx-auto">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && <ChatMessageLoading />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="px-4 py-2 bg-red-900/50 border-t border-red-800 text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Input area */}
      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-md">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-fractal-500/10 mb-4">
          <Brain className="w-8 h-8 text-fractal-500" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">
          Welcome to Fractal-Mind
        </h2>
        <p className="text-gray-400 mb-6">
          Your AI with evolutionary memory. Ask questions, share knowledge, and watch it learn.
        </p>
        <div className="grid grid-cols-1 gap-2 text-sm">
          <SuggestionChip icon={<Sparkles className="w-4 h-4" />}>
            What can you help me with?
          </SuggestionChip>
          <SuggestionChip icon={<Sparkles className="w-4 h-4" />}>
            Tell me about your memory system
          </SuggestionChip>
          <SuggestionChip icon={<Sparkles className="w-4 h-4" />}>
            How do you learn new things?
          </SuggestionChip>
        </div>
      </div>
    </div>
  );
}

interface SuggestionChipProps {
  icon: React.ReactNode;
  children: React.ReactNode;
}

function SuggestionChip({ icon, children }: SuggestionChipProps) {
  const { sendMessage } = useChatStore();
  const { namespace } = useSettingsStore();

  return (
    <button
      onClick={() => sendMessage(children as string, namespace)}
      className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-gray-300 hover:text-white transition-colors"
    >
      {icon}
      {children}
    </button>
  );
}
