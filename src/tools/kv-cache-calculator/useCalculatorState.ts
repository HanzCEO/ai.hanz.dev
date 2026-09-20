import { useMemo } from 'react'

import { DTYPES, type DtypeId } from '@/lib/kvcache'
import {
  CONFIG_SOURCE_DEFAULTS,
  CONFIG_SOURCE_SCHEMA,
  type ConfigSourceInputs,
} from '@/lib/use-config-source'
import { digitsOrNull, enumGuard, useUrlSyncedState, type UrlSchema } from '@/lib/url-state'

/**
 * The KV cache calculator inputs.
 *
 * The model source fields come from the shared schema, so this page and the
 * inference page describe a model in exactly the same way. Only the cache
 * fields belong to this page.
 */
export interface CalculatorInputs extends ConfigSourceInputs {
  /** Kept as text so the field can be emptied while typing. */
  contextLength: string
  sequenceCount: string
  kvCacheDtype: DtypeId
  indexerDtype: DtypeId
}

export const DEFAULT_MODEL_ID = CONFIG_SOURCE_DEFAULTS.modelId
export const DEFAULT_CONTEXT_LENGTH = '32768'
export const DEFAULT_SEQUENCE_COUNT = '1'

const DTYPE_IDS = DTYPES.map((dtype) => dtype.id)

const isDtype = enumGuard(DTYPE_IDS)

/**
 * Inputs live in the query string so a calculation can be shared or reloaded.
 * Declared once at module scope so the hook can capture it in a ref, and
 * exported so the URL contract can be tested without rendering the page.
 */
export const CALCULATOR_SCHEMA: UrlSchema<CalculatorInputs> = {
  ...CONFIG_SOURCE_SCHEMA,
  contextLength: { param: 'context', default: DEFAULT_CONTEXT_LENGTH, parse: digitsOrNull },
  sequenceCount: { param: 'sequences', default: DEFAULT_SEQUENCE_COUNT, parse: digitsOrNull },
  kvCacheDtype: {
    param: 'kv_dtype',
    default: 'BF16',
    parse: (raw) => (isDtype(raw) ? raw : null),
  },
  indexerDtype: {
    param: 'indexer_dtype',
    default: 'BF16',
    parse: (raw) => (isDtype(raw) ? raw : null),
  },
}

/** Which fields the URL supplied, under the names this page has always used. */
export interface CalculatorSeeded {
  context: boolean
  model: boolean
  dtype: boolean
}

export function useCalculatorState() {
  const { values: inputs, update, serialized, seeded: fieldSeeded } =
    useUrlSyncedState<CalculatorInputs>(CALCULATOR_SCHEMA)

  // The page checks seeded.current.context and seeded.current.dtype before it
  // applies its own defaults, so those two names are preserved here.
  const seeded = useMemo(
    () => ({
      get current(): CalculatorSeeded {
        const flags = fieldSeeded.current
        return {
          context: flags.contextLength,
          model: flags.modelId,
          dtype: flags.kvCacheDtype || flags.indexerDtype,
        }
      },
    }),
    [fieldSeeded],
  )

  return { inputs, update, serialized, seeded }
}
