import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/index.css';
import { initI18n } from '@/i18n';
import { useSettingsStore } from '@/stores/settingsStore';

async function bootstrap(): Promise<void> {
  const initialLanguage = useSettingsStore.getState().language;
  await initI18n(initialLanguage);
  await useSettingsStore.getState().initializeLanguage();

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

void bootstrap();
