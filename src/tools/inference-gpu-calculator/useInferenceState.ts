import { useMemo } from 'react'

import {
  DEFAULT_HEADROOM,
  DEFAULT_MTP_HEAD,
  DEFAULT_PRECISION,
  MAX_SUGGESTED_GPUS,
  MTP_HEAD_IDS,
  PRECISIONS,
  type InferencePrecision,
  type MtpHeadType,
} from '@/lib/inference'
import { MANUAL_DEFAULTS } from '@/lib/model-config'
import { decimalOrNull, digitsOrNull, enumOf, useUrlSyncedState, type UrlSchema } from '@/lib/url-state'
import {
  CALCULATOR_SCHEMA,
  type CalculatorInputs,
} from '@/tools/kv-cache-calculator/useCalculatorState'

/**
 * The inference page inputs.
 *
 * The model source, the context length, the sequence count and the cache dtypes
 * belong to the cache layer, so they come from that schema. Only the four
 * hardware fields are added here, which is what keeps the two layers from
 * offering the same input twice.
 */
export interface InferenceFormInputs extends CalculatorInputs {
  precision: InferencePrecision
  /** The speculative decoding head the model is served with. */
  mtpHead: MtpHeadType
  /** Held as a percentage so the field reads the way a person would write it. */
  headroomPercent: string
  maxGpus: string
}

/**
 * Qwen3-8B, so manual entry opens on a real model rather than on zeros.
 *
 * Declared once, beside the manual reader that consumes it, and re-exported
 * here so the page does not have to reach into two modules for one form.
 */
export { MANUAL_DEFAULTS }

export { DEFAULT_MODEL_ID } from '@/lib/use-config-source'

/** The default VRAM headroom, as the percentage the field shows. */
const DEFAULT_HEADROOM_PERCENT = String(DEFAULT_HEADROOM * 100)

/**
 * The four hardware fields this layer owns, layered over the cache schema.
 *
 * Spread rather than repeated, so the model source, the context length, the
 * sequence count and the cache dtypes are declared exactly once for the whole
 * page.
 */
export const INFERENCE_SCHEMA: UrlSchema<InferenceFormInputs> = {
  ...CALCULATOR_SCHEMA,
  precision: { param: 'precision', default: DEFAULT_PRECISION, parse: enumOf(PRECISIONS) },
  mtpHead: { param: 'mtp', default: DEFAULT_MTP_HEAD, parse: enumOf(MTP_HEAD_IDS) },
  headroomPercent: {
    param: 'headroom',
    default: DEFAULT_HEADROOM_PERCENT,
    parse: decimalOrNull,
  },
  maxGpus: { param: 'max_gpus', default: String(MAX_SUGGESTED_GPUS), parse: digitsOrNull },
}

/** Fields the URL supplied, under the names the page reads. */
export interface InferenceSeeded {
  model: boolean
  precision: boolean
  /** True when the MTP head came from the URL. */
  mtp: boolean
  context: boolean
  sequences: boolean
  /** True when either cache dtype came from the URL. */
  dtype: boolean
  headroom: boolean
  maxGpus: boolean
}

export function useInferenceState() {
  const { values: inputs, update, serialized, seeded: fieldSeeded } =
    useUrlSyncedState<InferenceFormInputs>(INFERENCE_SCHEMA)

  const seeded = useMemo(
    () => ({
      get current(): InferenceSeeded {
        const flags = fieldSeeded.current
        return {
          model: flags.modelId,
          precision: flags.precision,
          mtp: flags.mtpHead,
          context: flags.contextLength,
          sequences: flags.sequenceCount,
          dtype: flags.kvCacheDtype || flags.indexerDtype,
          headroom: flags.headroomPercent,
          maxGpus: flags.maxGpus,
        }
      },
    }),
    [fieldSeeded],
  )

  return { inputs, update, serialized, seeded }
}
