import { useMemo } from 'react'

import type { Provider } from '@/lib/kvcache'
import { parseTokenAmount } from '@/lib/token-size'
import { enumOf, nonEmptyText, useUrlSyncedState, type UrlSchema } from '@/lib/url-state'

/**
 * The calculator inputs.
 *
 * The provider and model id mirror the shared model source fields, but this
 * page reads a tokenizer.json rather than a config.json, so it does not offer
 * the paste or manual modes the size calculators need. Only the token count and
 * an optional hub token are this page's own.
 */
export interface TokenSizeInputs {
  provider: Provider
  modelId: string
  /** A hub token. Held in memory only. */
  token: string
  /** Kept as text so the field can be emptied while typing. */
  tokens: string
}

export const DEFAULT_TOKEN_SIZE_MODEL = 'Qwen/Qwen3-8B'
export const DEFAULT_TOKEN_AMOUNT = '1000000000'

/** Reads a token amount the way parseTokenAmount does, and keeps the raw text. */
function tokenAmountText(raw: string | null): string | null {
  if (raw === null) return null
  return parseTokenAmount(raw) === null ? null : raw.trim()
}

/**
 * Inputs live in the query string so a calculation can be shared or reloaded.
 * Declared once at module scope so the hook can capture it in a ref, and
 * exported so the URL contract can be tested without rendering the page.
 */
export const TOKEN_SIZE_SCHEMA: UrlSchema<TokenSizeInputs> = {
  provider: {
    param: 'provider',
    default: 'huggingface',
    parse: enumOf(['huggingface', 'modelscope']),
  },
  modelId: { param: 'model', default: DEFAULT_TOKEN_SIZE_MODEL, parse: nonEmptyText },
  token: { param: 'token', default: '', parse: () => null, omit: true },
  tokens: { param: 'tokens', default: DEFAULT_TOKEN_AMOUNT, parse: tokenAmountText },
}

export function useTokenSizeState() {
  const { values: inputs, update, serialized } = useUrlSyncedState<TokenSizeInputs>(TOKEN_SIZE_SCHEMA)

  return useMemo(() => ({ inputs, update, serialized }), [inputs, update, serialized])
}