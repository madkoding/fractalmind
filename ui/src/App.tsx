import { useState } from 'react';
import { Sidebar, ChatArea, SettingsModal, ModelManager } from '@/components';
import { VibeKanbanWebCompanion } from 'vibe-kanban-web-companion';

function App() {
  const [showSettings, setShowSettings] = useState(false);
  const [currentView, setCurrentView] = useState<'chat' | 'models'>('chat');

  return (
    <>
      <VibeKanbanWebCompanion />
      <div className="flex h-screen bg-gray-900 text-white">
        {/* Sidebar */}
        <Sidebar 
          onSettingsClick={() => setShowSettings(true)}
          onViewChange={setCurrentView}
          currentView={currentView}
        />

        {/* Main content */}
        {currentView === 'chat' ? <ChatArea /> : <ModelManager />}

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
