import type { Message } from '@/types';
import { User, Bot } from 'lucide-react';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useTranslation } from 'react-i18next';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const { t, i18n } = useTranslation();
  const isUser = message.role === 'user';

  // For new messages: latency_ms is stored on the message
  // For legacy messages (pre-migration): detect a locale-agnostic pattern like "(... NNN ms)"
  const legacyLatencyMatch =
    message.latency_ms === undefined
      ? message.content.match(/\(.*?(\d+) ms\)$/)
      : null;
  const legacyLatencyMs = legacyLatencyMatch ? parseInt(legacyLatencyMatch[1], 10) : undefined;

  const latencyMs = message.latency_ms ?? legacyLatencyMs;

  // Strip the legacy latency annotation from the content if present
  const displayContent =
    legacyLatencyMatch
      ? message.content.replace(/\n*\(.*?\d+ ms\)$/, '').trimEnd()
      : message.content;

  return (
    <div
      className={clsx(
        'flex gap-3 p-4 animate-in',
        isUser ? 'bg-gray-800/50' : 'bg-gray-900/50'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center',
          isUser ? 'bg-fractal-600' : 'bg-gray-700'
        )}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-fractal-400" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm text-gray-300">
            {isUser ? t('chat.you') : t('chat.assistant')}
          </span>
          <span className="text-xs text-gray-500">
            {new Date(message.timestamp).toLocaleTimeString(i18n.language)}
          </span>
          {message.role === 'assistant' && latencyMs !== undefined && (
            <span className="ml-2 text-xs text-gray-400">{t('chat.resolvedIn', { ms: latencyMs })}</span>
          )}
        </div>
        {isUser ? (
          <div className="text-gray-200 whitespace-pre-wrap break-words">
            {message.content}
          </div>
        ) : (
          <div className="text-gray-200 prose prose-invert prose-sm max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Headers
                h1: ({ children }) => (
                  <h1 className="text-xl font-bold text-white mt-4 mb-2">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-lg font-bold text-white mt-3 mb-2">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-base font-semibold text-white mt-2 mb-1">{children}</h3>
                ),
                // Paragraphs
                p: ({ children }) => (
                  <p className="mb-2 leading-relaxed">{children}</p>
                ),
                // Lists
                ul: ({ children }) => (
                  <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>
                ),
                li: ({ children }) => (
                  <li className="text-gray-200">{children}</li>
                ),
                // Code blocks
                code: ({ className, children }) => {
                  const isInline = !className;
                  if (isInline) {
                    return (
                      <code className="bg-gray-700 text-fractal-300 px-1.5 py-0.5 rounded text-sm font-mono">
                        {children}
                      </code>
                    );
                  }
                  return (
                    <code className="block bg-gray-800 text-gray-200 p-3 rounded-lg text-sm font-mono overflow-x-auto my-2">
                      {children}
                    </code>
                  );
                },
                pre: ({ children }) => (
                  <pre className="bg-gray-800 rounded-lg overflow-x-auto my-2">
                    {children}
                  </pre>
                ),
                // Links
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-fractal-400 hover:text-fractal-300 underline"
                  >
                    {children}
                  </a>
                ),
                // Blockquotes
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-fractal-500 pl-4 py-1 my-2 text-gray-300 italic">
                    {children}
                  </blockquote>
                ),
                // Tables
                table: ({ children }) => (
                  <div className="overflow-x-auto my-2">
                    <table className="min-w-full border border-gray-700 rounded">
                      {children}
                    </table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="bg-gray-800">{children}</thead>
                ),
                th: ({ children }) => (
                  <th className="px-3 py-2 text-left text-sm font-semibold text-gray-200 border-b border-gray-700">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="px-3 py-2 text-sm text-gray-300 border-b border-gray-700">
                    {children}
                  </td>
                ),
                // Horizontal rule
                hr: () => <hr className="border-gray-700 my-4" />,
                // Strong and emphasis
                strong: ({ children }) => (
                  <strong className="font-bold text-white">{children}</strong>
                ),
                em: ({ children }) => (
                  <em className="italic text-gray-300">{children}</em>
                ),
              }}
            >
              {displayContent}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

// Loading indicator for streaming responses
export function ChatMessageLoading() {
  const { t } = useTranslation();

  return (
    <div className="flex gap-3 p-4 bg-gray-900/50 animate-in">
      <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gray-700 flex items-center justify-center">
        <Bot className="w-5 h-5 text-fractal-400" />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm text-gray-300">{t('chat.assistant')}</span>
          <span className="text-xs text-gray-500">{t('chat.loading')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-fractal-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 bg-fractal-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 bg-fractal-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}
