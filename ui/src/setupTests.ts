import '@testing-library/jest-dom';
import { beforeAll } from 'vitest';
import { initI18n } from '@/i18n/config';

beforeAll(async () => {
  await initI18n('en');
});
