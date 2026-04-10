const ERROR_CODE_MAP: Record<string, string> = {
  'ERR_TEC_EMB_001': 'error.embeddingFailed',
  'ERR_TEC_EMB_002': 'error.embeddingTimeout',
  'ERR_TEC_EMB_UNKNOWN': 'error.embeddingUnknown',
  'ERR_TEC_LLM_001': 'error.llmRequestFailed',
  'ERR_TEC_LLM_002': 'error.llmParseFailed',
  'ERR_TEC_LLM_003': 'error.llmTimeout',
  'ERR_TEC_LLM_UNKNOWN': 'error.llmUnknown',
  'ERR_TEC_DB_001': 'error.dbQueryFailed',
  'ERR_TEC_DB_002': 'error.dbConnectionFailed',
  'ERR_TEC_DB_UNKNOWN': 'error.dbUnknown',
  'ERR_TEC_UNKNOWN': 'error.technicalUnknown',
  'ERR_VAL_EMPTY_CONTENT': 'error.emptyContent',
  'ERR_VAL_EMPTY_QUESTION': 'error.emptyQuestion',
  'ERR_VAL_QUESTION_TOO_LONG': 'error.questionTooLong',
  'ERR_VAL_UNKNOWN': 'error.validationUnknown',
  'ERR_UNKNOWN': 'error.unknown',
  'ERR_BIZ_EMPTY_KB': 'error.emptyKnowledgeBase',
  'ERR_BIZ_NO_RESULTS': 'error.noSearchResults',
  'ERR_SVC_UNAVAILABLE': 'error.serviceUnavailable',
  'ERR_SVC_UNKNOWN': 'error.serviceUnknown',
};

export function parseApiError(error: unknown): string {
  if (error instanceof Error) {
    const msg = error.message;
    
    if (msg.startsWith('API Error:')) {
      try {
        const json = JSON.parse(msg.replace(/^API Error: \d+ - /, ''));
        if (json.error?.code && json.error?.message) {
          const key = ERROR_CODE_MAP[json.error.code];
          return key ? `errors.${key}` : json.error.message;
        }
      } catch {
        const text = msg.replace(/^API Error: \d+ - /, '');
        if (text.length < 200) return text;
      }
    }
    
    return msg;
  }
  return 'errors.unknown';
}

export function getErrorMessage(error: unknown, t: (key: string) => string): string {
  const parsed = parseApiError(error);
  if (parsed.startsWith('errors.')) {
    return t(parsed);
  }
  return parsed;
}