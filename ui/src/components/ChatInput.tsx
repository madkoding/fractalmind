import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip } from 'lucide-react';
import IngestFileUploader from './IngestFileUploader';
import clsx from 'clsx';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Ask anything...',
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [showUploader, setShowUploader] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = () => {
    const trimmed = message.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setMessage('');
      // Reset height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-700 p-4 bg-gray-800">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end gap-2 bg-gray-900 rounded-xl border border-gray-700 focus-within:border-fractal-500 transition-colors">
          {/* Attachment button (placeholder) */}
          <button
            type="button"
            className="p-3 text-gray-400 hover:text-white transition-colors"
            title="Attach file"
            onClick={() => setShowUploader((s) => !s)}
            data-testid="toggle-uploader"
          >
            <Paperclip className="w-5 h-5" />
          </button>

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className={clsx(
              'flex-1 bg-transparent py-3 resize-none outline-none text-white placeholder-gray-500',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          />

          {/* Send button */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={disabled || !message.trim()}
            className={clsx(
              'p-3 rounded-lg transition-colors',
              message.trim() && !disabled
                ? 'text-fractal-500 hover:bg-fractal-500/10'
                : 'text-gray-600 cursor-not-allowed'
            )}
            title="Send message"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>

        {/* Uploader (toggle) */}
        {showUploader && <div className="mt-3"><IngestFileUploader /></div>}

        {/* Helper text */}
        <p className="text-xs text-gray-500 mt-2 text-center">
          Press <kbd className="px-1 py-0.5 bg-gray-700 rounded">Enter</kbd> to send,{' '}
          <kbd className="px-1 py-0.5 bg-gray-700 rounded">Shift+Enter</kbd> for new line
        </p>
      </div>
    </div>
  );
}
