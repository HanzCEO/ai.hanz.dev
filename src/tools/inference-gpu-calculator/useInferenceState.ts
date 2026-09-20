import { useMemo } from 'react'

import {
  DEFAULT_CONTEXT_LENGTH,
  DEFAULT_HEADROOM,
  DEFAULT_PRECISION,
  DEFAULT_SEQUENCES,
  MAX_SUGGESTED_GPUS,
  PRECISIONS,
  type InferencePrecision,
} from '@/lib/inference'
import type { Provider } from '@/lib/kvcache'
import {
  decimalOrNull,
  digitsOrNull,
  enumOf,
  nonEmptyText,
  useUrlSyncedState,
  type UrlSchema,
} from '@/lib/url-state'

/** Where the model shape comes from. */
export type InputMode = 'hub' | 'paste' | 'manual'

export interface InferenceFormInputs {
  mode: InputMode

  // Hub mode.
  provider: Provider
  modelId: string
  /** Never written to the URL. */
  token: string

  // Paste mode. Kept out of the URL, since a config can be long.
  configText: string

  // Manual mode. The values are written back into a config shape so the same
  // detector reads every input mode.
  hiddenSize: string
  intermediateSize: string
  numLayers: string
  vocabSize: string
  attentionHeads: string
  kvHeads: string
  headDim: string
  routedExperts: string
  expertsPerToken: string
  moeIntermediateSize: string
  moeLayers: string
  /**
   * Whether the embedding and the language model head share one table.
   *
   * This value moves the weight figure by one embedding table, which is more
   * than a gibibyte on an 8B model. The config carries it in every other mode,
   * so manual entry has to be able to state it as well.
   */
  tieEmbeddings: 'tied' | 'untied'

  // What is served.
  precision: InferencePrecision
  contextLength: string
  sequences: string

  // Hardware limits.
  /** Held as a percentage so the field reads the way a person would write it. */
  headroomPercent: string
  maxGpus: string
}

/**
 * Qwen3-8B, so manual entry opens on a real model rather than on zeros.
 *
 * hidden 4096, 36 blocks, a dense width of 12288, a vocabulary of 151936, 32
 * attention heads and 8 key and value heads at a head width of 128, untied.
 */
export const MANUAL_DEFAULTS = {
  hiddenSize: '4096',
  intermediateSize: '12288',
  numLayers: '36',
  vocabSize: '151936',
  attentionHeads: '32',
  kvHeads: '8',
  headDim: '128',
  routedExperts: '0',
  expertsPerToken: '0',
  moeIntermediateSize: '0',
  moeLayers: '0',
  tieEmbeddings: 'untied' as const,
}

export const DEFAULT_MODEL_ID = 'Qwen/Qwen3-8B'

const DEFAULTS: InferenceFormInputs = {
  mode: 'hub',
  provider: 'huggingface',
  modelId: DEFAULT_MODEL_ID,
  token: '',
  configText: '',
  ...MANUAL_DEFAULTS,
  precision: DEFAULT_PRECISION,
  contextLength: String(DEFAULT_CONTEXT_LENGTH),
  sequences: String(DEFAULT_SEQUENCES),
  headroomPercent: String(DEFAULT_HEADROOM * 100),
  maxGpus: String(MAX_SUGGESTED_GPUS),
}

const MODES: InputMode[] = ['hub', 'paste', 'manual']
const TIE_MODES: Array<InferenceFormInputs['tieEmbeddings']> = ['untied', 'tied']

export const INFERENCE_SCHEMA: UrlSchema<InferenceFormInputs> = {
  mode: { param: 'mode', default: DEFAULTS.mode, parse: enumOf(MODES) },
  provider: {
    param: 'provider',
    default: DEFAULTS.provider,
    parse: enumOf(['huggingface', 'modelscope']),
  },
  modelId: { param: 'model', default: DEFAULTS.modelId, parse: nonEmptyText },
  token: { param: 'token', default: '', parse: () => null, omit: true },
  configText: { param: 'config', default: '', parse: () => null, omit: true },

  hiddenSize: { param: 'hidden', default: DEFAULTS.hiddenSize, parse: digitsOrNull },
  intermediateSize: { param: 'ffn', default: DEFAULTS.intermediateSize, parse: digitsOrNull },
  numLayers: { param: 'layers', default: DEFAULTS.numLayers, parse: digitsOrNull },
  vocabSize: { param: 'vocab', default: DEFAULTS.vocabSize, parse: digitsOrNull },
  attentionHeads: { param: 'heads', default: DEFAULTS.attentionHeads, parse: digitsOrNull },
  kvHeads: { param: 'kv_heads', default: DEFAULTS.kvHeads, parse: digitsOrNull },
  headDim: { param: 'head_dim', default: DEFAULTS.headDim, parse: digitsOrNull },
  routedExperts: { param: 'experts', default: DEFAULTS.routedExperts, parse: digitsOrNull },
  expertsPerToken: { param: 'topk', default: DEFAULTS.expertsPerToken, parse: digitsOrNull },
  moeIntermediateSize: {
    param: 'expert_ffn',
    default: DEFAULTS.moeIntermediateSize,
    parse: digitsOrNull,
  },
  moeLayers: { param: 'moe_layers', default: DEFAULTS.moeLayers, parse: digitsOrNull },
  tieEmbeddings: { param: 'tied', default: DEFAULTS.tieEmbeddings, parse: enumOf(TIE_MODES) },

  precision: { param: 'precision', default: DEFAULTS.precision, parse: enumOf(PRECISIONS) },
  contextLength: { param: 'context', default: DEFAULTS.contextLength, parse: digitsOrNull },
  sequences: { param: 'sequences', default: DEFAULTS.sequences, parse: digitsOrNull },

  headroomPercent: { param: 'headroom', default: DEFAULTS.headroomPercent, parse: decimalOrNull },
  maxGpus: { param: 'max_gpus', default: DEFAULTS.maxGpus, parse: digitsOrNull },
}

/** Fields the URL supplied, under the names the page reads. */
export interface InferenceSeeded {
  model: boolean
  precision: boolean
  context: boolean
  sequences: boolean
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
          context: flags.contextLength,
          sequences: flags.sequences,
          headroom: flags.headroomPercent,
          maxGpus: flags.maxGpus,
        }
      },
    }),
    [fieldSeeded],
  )

  return { inputs, update, serialized, seeded }
}
