import { useMemo } from 'react'

import { GPU_PRESETS, STORAGE_PRESETS } from '@/lib/hardware'
import type { Provider } from '@/lib/kvcache'
import type { WeightDtype } from '@/lib/reap'
import { useUrlSyncedState, type UrlSchema } from '@/lib/url-state'

/** Where the model shape comes from. */
export type InputMode = 'hub' | 'paste' | 'manual'

export interface ReapFormInputs {
  mode: InputMode

  // Hub mode.
  provider: Provider
  modelId: string
  /** Never written to the URL. */
  token: string

  // Paste mode. Kept out of the URL, since a config can be long.
  configText: string

  // Manual mode. The values are turned back into a config shape so the same
  // detector reads every input mode.
  hiddenSize: string
  intermediateSize: string
  moeIntermediateSize: string
  routedExperts: string
  expertsPerToken: string
  numLayers: string
  moeLayers: string
  sharedExperts: string
  sharedExpertIntermediate: string
  vocabSize: string
  attentionHeads: string
  headDim: string
  kvHeads: string

  // Calibration.
  samples: string
  sequenceLength: string

  // Pruning and hardware.
  pruneRatio: string
  gpuId: string
  storageId: string
  weightDtype: WeightDtype

  // Advanced.
  mfu: string
  overheadFactor: string
  microBatchSize: string
  scaleTopK: boolean
}

/**
 * The Qwen3-30B-A3B shape, which is the model the published REAP example
 * calibrates. Manual entry opens on it so the fields show a real mixture of
 * experts rather than zeros.
 */
export const MANUAL_DEFAULTS = {
  hiddenSize: '2048',
  intermediateSize: '6144',
  moeIntermediateSize: '768',
  routedExperts: '128',
  expertsPerToken: '8',
  numLayers: '48',
  moeLayers: '48',
  sharedExperts: '0',
  sharedExpertIntermediate: '768',
  vocabSize: '151936',
  attentionHeads: '32',
  headDim: '128',
  kvHeads: '4',
}

export const DEFAULT_MODEL_ID = 'Qwen/Qwen3-30B-A3B-Instruct-2507'

const DEFAULTS: ReapFormInputs = {
  mode: 'hub',
  provider: 'huggingface',
  modelId: DEFAULT_MODEL_ID,
  token: '',
  configText: '',
  ...MANUAL_DEFAULTS,
  samples: '512',
  sequenceLength: '2048',
  pruneRatio: '0.4',
  gpuId: 'rtx-5090',
  storageId: 'nvme-pcie4',
  weightDtype: 'BF16',
  mfu: '0.3',
  overheadFactor: '2',
  microBatchSize: '1',
  scaleTopK: false,
}

const GPU_IDS = GPU_PRESETS.map((gpu) => gpu.id)
const STORAGE_IDS = STORAGE_PRESETS.map((storage) => storage.id)
const WEIGHT_DTYPES: WeightDtype[] = ['BF16', 'FP8', 'INT4']
const MODES: InputMode[] = ['hub', 'paste', 'manual']

function enumOf<T extends string>(allowed: readonly T[]) {
  return (raw: string | null): T | null =>
    raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : null
}

function digitsOrNull(raw: string | null): string | null {
  if (raw === null) return null
  const trimmed = raw.trim()
  return /^\d+$/.test(trimmed) ? trimmed : null
}

function decimalOrNull(raw: string | null): string | null {
  if (raw === null) return null
  const trimmed = raw.trim()
  return /^\d+(\.\d+)?$/.test(trimmed) ? trimmed : null
}

const REAP_SCHEMA: UrlSchema<ReapFormInputs> = {
  mode: { param: 'mode', default: DEFAULTS.mode, parse: enumOf(MODES) },
  provider: { param: 'provider', default: DEFAULTS.provider, parse: enumOf(['huggingface', 'modelscope']) },
  modelId: {
    param: 'model',
    default: DEFAULTS.modelId,
    parse: (raw) => {
      if (raw === null) return null
      const trimmed = raw.trim()
      return trimmed === '' ? null : trimmed
    },
  },
  token: { param: 'token', default: '', parse: () => null, omit: true },
  configText: { param: 'config', default: '', parse: () => null, omit: true },

  hiddenSize: { param: 'hidden', default: DEFAULTS.hiddenSize, parse: digitsOrNull },
  intermediateSize: { param: 'ffn', default: DEFAULTS.intermediateSize, parse: digitsOrNull },
  moeIntermediateSize: {
    param: 'expert_ffn',
    default: DEFAULTS.moeIntermediateSize,
    parse: digitsOrNull,
  },
  routedExperts: { param: 'experts', default: DEFAULTS.routedExperts, parse: digitsOrNull },
  expertsPerToken: { param: 'topk', default: DEFAULTS.expertsPerToken, parse: digitsOrNull },
  numLayers: { param: 'layers', default: DEFAULTS.numLayers, parse: digitsOrNull },
  moeLayers: { param: 'moe_layers', default: DEFAULTS.moeLayers, parse: digitsOrNull },
  sharedExperts: { param: 'shared', default: DEFAULTS.sharedExperts, parse: digitsOrNull },
  sharedExpertIntermediate: {
    param: 'shared_ffn',
    default: DEFAULTS.sharedExpertIntermediate,
    parse: digitsOrNull,
  },
  vocabSize: { param: 'vocab', default: DEFAULTS.vocabSize, parse: digitsOrNull },
  attentionHeads: { param: 'heads', default: DEFAULTS.attentionHeads, parse: digitsOrNull },
  headDim: { param: 'head_dim', default: DEFAULTS.headDim, parse: digitsOrNull },
  kvHeads: { param: 'kv_heads', default: DEFAULTS.kvHeads, parse: digitsOrNull },

  samples: { param: 'samples', default: DEFAULTS.samples, parse: digitsOrNull },
  sequenceLength: { param: 'seq', default: DEFAULTS.sequenceLength, parse: digitsOrNull },

  pruneRatio: { param: 'ratio', default: DEFAULTS.pruneRatio, parse: decimalOrNull },
  gpuId: { param: 'gpu', default: DEFAULTS.gpuId, parse: enumOf(GPU_IDS) },
  storageId: { param: 'storage', default: DEFAULTS.storageId, parse: enumOf(STORAGE_IDS) },
  weightDtype: {
    param: 'dtype',
    default: DEFAULTS.weightDtype,
    parse: enumOf(WEIGHT_DTYPES),
  },

  mfu: { param: 'mfu', default: DEFAULTS.mfu, parse: decimalOrNull },
  overheadFactor: { param: 'overhead', default: DEFAULTS.overheadFactor, parse: decimalOrNull },
  microBatchSize: { param: 'micro_batch', default: DEFAULTS.microBatchSize, parse: digitsOrNull },
  scaleTopK: {
    param: 'scale_topk',
    default: DEFAULTS.scaleTopK,
    parse: (raw) => (raw === '1' ? true : raw === '0' ? false : null),
    serialize: (value) => (value ? '1' : '0'),
  },
}

/** Fields the URL supplied, under the names the page reads. */
export interface ReapSeeded {
  model: boolean
  dtype: boolean
  gpu: boolean
}

export function useReapState() {
  const { values: inputs, update, serialized, seeded: fieldSeeded } =
    useUrlSyncedState<ReapFormInputs>(REAP_SCHEMA)

  const seeded = useMemo(
    () => ({
      get current(): ReapSeeded {
        const flags = fieldSeeded.current
        return {
          model: flags.modelId,
          dtype: flags.weightDtype,
          gpu: flags.gpuId,
        }
      },
    }),
    [fieldSeeded],
  )

  return { inputs, update, serialized, seeded }
}
