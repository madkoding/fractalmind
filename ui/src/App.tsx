import { useState } from 'react';
import { Sidebar, ChatArea, SettingsModal } from '@/components';
import { VibeKanbanWebCompanion } from 'vibe-kanban-web-companion';
import { Menu, X } from 'lucide-react';

function App() {
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <>
      <VibeKanbanWebCompanion />
      <div className="flex h-screen bg-gray-900 text-white">
        {/* Mobile sidebar backdrop */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar */}
        <div
          className={`
            fixed lg:static inset-y-0 left-0 z-50
            transform transition-transform duration-300 ease-in-out
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
            w-64 bg-gray-800 flex flex-col h-full border-r border-gray-700
          `}
        >
          <button
            onClick={() => setSidebarOpen(false)}
            className="absolute top-4 right-4 p-2 lg:hidden hover:bg-gray-700 rounded-lg"
          >
            <X className="w-5 h-5" />
          </button>
          <Sidebar onSettingsClick={() => setShowSettings(true)} />
        </div>

        {/* Main content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Mobile header */}
          <div className="lg:hidden flex items-center p-4 border-b border-gray-700">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-gray-700 rounded-lg mr-2"
            >
              <Menu className="w-6 h-6" />
            </button>
            <span className="text-lg font-bold">Fractal-Mind</span>
          </div>

          <ChatArea />
        </div>

        {/* Settings Modal */}
        <SettingsModal
          isOpen={showSettings}
          onClose={() => setShowSettings(false)}
        />
      </div>
    </>
  );
}

export default App;
