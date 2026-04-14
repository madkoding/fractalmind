import { afterEach, describe, expect, it, vi } from 'vitest';
import { detectLanguageByIp, detectPreferredLanguage } from './detectLanguage';

function mockNavigatorLanguage(language?: string, languages?: string[]): void {
  Object.defineProperty(window.navigator, 'language', {
    configurable: true,
    value: language,
  });

  Object.defineProperty(window.navigator, 'languages', {
    configurable: true,
    value: languages,
  });
}

describe('language detection', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('uses locale from geoip response first', async () => {
    vi.spyOn(window, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        country_code: 'US',
        language: 'pt-BR',
      }),
    } as Response);

    const detected = await detectLanguageByIp();
    expect(detected).toBe('pt');
  });

  it('maps country when locale not present', async () => {
    vi.spyOn(window, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        country_code: 'JP',
      }),
    } as Response);

    const detected = await detectLanguageByIp();
    expect(detected).toBe('ja');
  });

  it('falls back to browser language when geoip fails', async () => {
    vi.spyOn(window, 'fetch').mockRejectedValue(new Error('network'));
    mockNavigatorLanguage('de-DE', ['de-DE', 'en-US']);

    const detected = await detectPreferredLanguage();
    expect(detected).toBe('de');
  });

  it('falls back to navigator.languages when language is unknown', async () => {
    vi.spyOn(window, 'fetch').mockResolvedValue({
      ok: false,
    } as Response);
    mockNavigatorLanguage('xx-YY', ['fr-FR']);

    const detected = await detectPreferredLanguage();
    expect(detected).toBe('fr');
  });

  it('defaults to english when no match exists', async () => {
    vi.spyOn(window, 'fetch').mockResolvedValue({
      ok: false,
    } as Response);
    mockNavigatorLanguage('xx-YY', ['zz-ZZ']);

    const detected = await detectPreferredLanguage();
    expect(detected).toBe('en');
  });
});
