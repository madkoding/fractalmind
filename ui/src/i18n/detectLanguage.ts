import type { SupportedLanguage } from '@/types';
import { DEFAULT_LANGUAGE, isSupportedLanguage } from './languages';

interface GeoIpResponse {
  country_code?: string;
  countryCode?: string;
  country?: string;
  countryCodeIso2?: string;
  language?: string;
  languages?: string;
  locale?: string;
}

const GEOIP_URL = import.meta.env.VITE_GEOIP_URL || 'https://ipapi.co/json/';

const COUNTRY_TO_LANGUAGE: Record<string, SupportedLanguage> = {
  AR: 'es',
  BO: 'es',
  CL: 'es',
  CO: 'es',
  CR: 'es',
  CU: 'es',
  DO: 'es',
  EC: 'es',
  ES: 'es',
  GT: 'es',
  HN: 'es',
  MX: 'es',
  NI: 'es',
  PA: 'es',
  PE: 'es',
  PR: 'es',
  PY: 'es',
  SV: 'es',
  UY: 'es',
  VE: 'es',
  BR: 'pt',
  PT: 'pt',
  FR: 'fr',
  BE: 'fr',
  CH: 'fr',
  DE: 'de',
  AT: 'de',
  LI: 'de',
  IT: 'it',
  RU: 'ru',
  BY: 'ru',
  KZ: 'ru',
  JP: 'ja',
  CN: 'zh-CN',
  SG: 'zh-CN',
  TW: 'zh-CN',
  HK: 'zh-CN',
  MO: 'zh-CN',
};

function normalizeBrowserLanguage(locale?: string | null): SupportedLanguage | null {
  if (!locale) {
    return null;
  }

  const normalized = locale.trim();
  if (isSupportedLanguage(normalized)) {
    return normalized;
  }

  const base = normalized.split('-')[0]?.toLowerCase();
  if (!base) {
    return null;
  }

  if (base === 'zh') {
    return 'zh-CN';
  }

  return isSupportedLanguage(base) ? base : null;
}

function mapCountryToLanguage(country?: string | null): SupportedLanguage | null {
  if (!country) {
    return null;
  }
  const upper = country.toUpperCase();
  return COUNTRY_TO_LANGUAGE[upper] || null;
}

export async function detectLanguageByIp(): Promise<SupportedLanguage | null> {
  try {
    const response = await fetch(GEOIP_URL, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as GeoIpResponse;
    const countryCode = data.country_code || data.countryCode || data.countryCodeIso2 || data.country;
    const localeLanguage = normalizeBrowserLanguage(data.language || data.locale || data.languages?.split(',')[0]);

    if (localeLanguage) {
      return localeLanguage;
    }

    return mapCountryToLanguage(countryCode);
  } catch {
    return null;
  }
}

export async function detectPreferredLanguage(): Promise<SupportedLanguage> {
  const fromIp = await detectLanguageByIp();
  if (fromIp) {
    return fromIp;
  }

  const fromBrowser = normalizeBrowserLanguage(navigator.language)
    || normalizeBrowserLanguage(navigator.languages?.[0]);
  if (fromBrowser) {
    return fromBrowser;
  }

  return DEFAULT_LANGUAGE;
}
