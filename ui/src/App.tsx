import { useState } from 'react';
import { Sidebar, ChatArea, SettingsModal } from '@/components';

function App() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <Sidebar onSettingsClick={() => setShowSettings(true)} />

      {/* Main content */}
      <ChatArea />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
}

export default App;
