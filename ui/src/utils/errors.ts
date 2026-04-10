const ERROR_CODE_MAP: Record<string, string> = {
  'ERR_TEC_EMB_001': 'errors.embeddingFailed',
  'ERR_TEC_EMB_002': 'errors.embeddingTimeout',
  'ERR_TEC_EMB_UNKNOWN': 'errors.embeddingUnknown',
  'ERR_TEC_LLM_001': 'errors.llmRequestFailed',
  'ERR_TEC_LLM_002': 'errors.llmParseFailed',
  'ERR_TEC_LLM_003': 'errors.llmTimeout',
  'ERR_TEC_LLM_UNKNOWN': 'errors.llmUnknown',
  'ERR_TEC_DB_001': 'errors.dbQueryFailed',
  'ERR_TEC_DB_002': 'errors.dbConnectionFailed',
  'ERR_TEC_DB_UNKNOWN': 'errors.dbUnknown',
  'ERR_TEC_UNKNOWN': 'errors.technicalUnknown',
  'ERR_VAL_EMPTY_CONTENT': 'errors.emptyContent',
  'ERR_VAL_EMPTY_QUESTION': 'errors.emptyQuestion',
  'ERR_VAL_QUESTION_TOO_LONG': 'errors.questionTooLong',
  'ERR_VAL_UNKNOWN': 'errors.validationUnknown',
  'ERR_UNKNOWN': 'errors.unknown',
  'ERR_BIZ_EMPTY_KB': 'errors.emptyKnowledgeBase',
  'ERR_BIZ_NO_RESULTS': 'errors.noSearchResults',
  'ERR_SVC_UNAVAILABLE': 'errors.serviceUnavailable',
  'ERR_SVC_UNKNOWN': 'errors.serviceUnavailable',
};

export function parseApiError(error: unknown): string {
  if (error instanceof Error) {
    const msg = error.message;
    
    if (msg.startsWith('API Error:')) {
      try {
        const json = JSON.parse(msg.replace(/^API Error: \d+ - /, ''));
        if (json.error?.code && json.error?.message) {
          const key = ERROR_CODE_MAP[json.error.code];
          return key ?? json.error.message;
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