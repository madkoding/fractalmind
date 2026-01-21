import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Message, Conversation, AskResponse } from '@/types';
import { api } from '@/services/api';

interface ChatState {
  conversations: Conversation[];
  currentConversationId: string | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  createConversation: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  sendMessage: (content: string, namespace?: string) => Promise<void>;
  clearError: () => void;
}

const generateId = () => crypto.randomUUID();

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      currentConversationId: null,
      isLoading: false,
      error: null,

      createConversation: () => {
        const id = generateId();
        const conversation: Conversation = {
          id,
          title: 'New Conversation',
          messages: [],
          created_at: Date.now(),
          updated_at: Date.now(),
        };

        set((state) => ({
          conversations: [conversation, ...state.conversations],
          currentConversationId: id,
        }));

        return id;
      },

      selectConversation: (id: string) => {
        set({ currentConversationId: id });
      },

      deleteConversation: (id: string) => {
        set((state) => {
          const conversations = state.conversations.filter((c) => c.id !== id);
          const currentConversationId =
            state.currentConversationId === id
              ? conversations[0]?.id || null
              : state.currentConversationId;

          return { conversations, currentConversationId };
        });
      },

      sendMessage: async (content: string, namespace?: string) => {
        const state = get();
        let conversationId = state.currentConversationId;

        // Create new conversation if none exists
        if (!conversationId) {
          conversationId = state.createConversation();
        }

        // Add user message
        const userMessage: Message = {
          id: generateId(),
          role: 'user',
          content,
          timestamp: Date.now(),
        };

        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  messages: [...c.messages, userMessage],
                  updated_at: Date.now(),
                  title: c.messages.length === 0 ? content.slice(0, 50) : c.title,
                }
              : c
          ),
          isLoading: true,
          error: null,
        }));

        try {
          // Call API
          const response: AskResponse = await api.ask({
            question: content,
            namespace,
          });

          // Add assistant message
          const assistantMessage: Message = {
            id: generateId(),
            role: 'assistant',
            content: response.answer,
            timestamp: Date.now(),
          };

          set((state) => ({
            conversations: state.conversations.map((c) =>
              c.id === conversationId
                ? {
                    ...c,
                    messages: [...c.messages, assistantMessage],
                    updated_at: Date.now(),
                  }
                : c
            ),
            isLoading: false,
          }));
        } catch (error) {
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Unknown error',
          });
        }
      },

      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'fractalmind-chat',
      partialize: (state) => ({
        conversations: state.conversations,
        currentConversationId: state.currentConversationId,
      }),
    }
  )
);

// Selectors
export const useCurrentConversation = () => {
  const { conversations, currentConversationId } = useChatStore();
  return conversations.find((c) => c.id === currentConversationId);
};

export const useMessages = () => {
  const conversation = useCurrentConversation();
  return conversation?.messages || [];
};
