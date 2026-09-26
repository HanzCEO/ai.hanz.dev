import { useMemo } from 'react'

import {
  DEFAULT_COMPRESSION_PERCENT,
  DEFAULT_PRICE_PRESET_ID,
  DEFAULT_WORKLOAD_PRESET,
  findPricePreset,
  findWorkloadPreset,
  type PricePreset,
  type WorkloadPreset,
} from '@/lib/compression'
import { decimalOrNull, useUrlSyncedState, type UrlSchema } from '@/lib/url-state'

/** Every field is held as text, so a half typed number does not become a value. */
export interface CompressionFormInputs {
  inputPrice: string
  cachedInputPrice: string
  outputPrice: string
  missPercent: string
  cachePercent: string
  outputPercent: string
  compressionPercent: string
}

const DEFAULT_PRICE = findPricePreset(DEFAULT_PRICE_PRESET_ID) as PricePreset
const DEFAULT_WORKLOAD = findWorkloadPreset(DEFAULT_WORKLOAD_PRESET) as WorkloadPreset

const DEFAULTS: CompressionFormInputs = {
  inputPrice: String(DEFAULT_PRICE.inputPrice),
  cachedInputPrice: String(DEFAULT_PRICE.cachedInputPrice),
  outputPrice: String(DEFAULT_PRICE.outputPrice),
  missPercent: String(DEFAULT_WORKLOAD.mix.missPercent),
  cachePercent: String(DEFAULT_WORKLOAD.mix.cachePercent),
  outputPercent: String(DEFAULT_WORKLOAD.mix.outputPercent),
  compressionPercent: String(DEFAULT_COMPRESSION_PERCENT),
}

export const COMPRESSION_SCHEMA: UrlSchema<CompressionFormInputs> = {  inputPrice: { param: 'in', default: DEFAULTS.inputPrice, parse: decimalOrNull },
  cachedInputPrice: { param: 'cin', default: DEFAULTS.cachedInputPrice, parse: decimalOrNull },
  outputPrice: { param: 'out', default: DEFAULTS.outputPrice, parse: decimalOrNull },
  missPercent: { param: 'miss', default: DEFAULTS.missPercent, parse: decimalOrNull },
  cachePercent: { param: 'cache', default: DEFAULTS.cachePercent, parse: decimalOrNull },
  outputPercent: { param: 'outtok', default: DEFAULTS.outputPercent, parse: decimalOrNull },
  compressionPercent: {
    param: 'comp',
    default: DEFAULTS.compressionPercent,
    parse: decimalOrNull,
  },
}

export function useCompressionState() {
  const { values: inputs, update, serialized } =
    useUrlSyncedState<CompressionFormInputs>(COMPRESSION_SCHEMA)

  return useMemo(() => ({ inputs, update, serialized }), [inputs, update, serialized])
}
