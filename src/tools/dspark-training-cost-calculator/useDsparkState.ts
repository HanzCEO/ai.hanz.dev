import { useMemo } from 'react'

import { GPU_PRESETS, STORAGE_PRESETS } from '@/lib/hardware'
import {
  DEFAULT_PRESET,
  DSPARK_PRESETS,
  type DsparkDataMode,
  type DsparkPreset,
} from '@/lib/dspark'
import type { Provider } from '@/lib/kvcache'
import {
  decimalOrNull,
  digitsOrNull,
  enumOf,
  nonEmptyText,
  useUrlSyncedState,
  type UrlSchema,
} from '@/lib/url-state'

/** Where the target shape comes from. */
export type InputMode = 'hub' | 'paste' | 'manual'

export interface DsparkFormInputs {
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

  // Training data.
  samples: string
  sequenceLength: string
  epochs: string

  // Recipe.
  dataMode: DsparkDataMode
  numTargetLayers: string
  numDraftLayers: string
  blockSize: string
  numAnchors: string
  markovRank: string

  // Hardware.
  gpuId: string
  gpuCount: string
  storageId: string

  // Advanced.
  mfu: string
  overheadFactor: string
  microBatchSize: string
}

/**
 * MiniCPM5-2B, the target the MiniCPM5-2B-DSpark recipe was trained against.
 * Manual entry opens on it so the fields show a real target rather than zeros,
 * and so the default view costs a run whose settings are all published.
 */
export const MANUAL_DEFAULTS = {
  hiddenSize: '2048',
  intermediateSize: '6144',
  numLayers: '42',
  vocabSize: '130560',
  attentionHeads: '16',
  kvHeads: '2',
  headDim: '128',
  routedExperts: '0',
  expertsPerToken: '0',
  moeIntermediateSize: '0',
  moeLayers: '0',
}

export const DEFAULT_MODEL_ID = 'openbmb/MiniCPM5-2B'

const OPENING_PRESET = DSPARK_PRESETS.find((preset) => preset.id === DEFAULT_PRESET) as DsparkPreset

const DEFAULTS: DsparkFormInputs = {
  // Manual entry opens by default, pre-filled with MiniCPM5-2B and the recipe
  // OpenBMB published for its drafter. A calculator should show a worked answer
  // before anything is typed, and it means the prerendered page carries real
  // figures rather than a spinner. Switching to a model id reads a config.
  mode: 'manual',
  provider: 'huggingface',
  modelId: DEFAULT_MODEL_ID,
  token: '',
  configText: '',
  ...MANUAL_DEFAULTS,
  samples: String(OPENING_PRESET.samples),
  sequenceLength: String(OPENING_PRESET.sequenceLength),
  epochs: String(OPENING_PRESET.epochs),
  dataMode: 'offline',
  numTargetLayers: '5',
  numDraftLayers: '5',
  blockSize: '7',
  numAnchors: String(OPENING_PRESET.numAnchors),
  markovRank: '256',
  gpuId: 'rtx-5090',
  gpuCount: '1',
  storageId: 'pcie5-host-ram',
  mfu: '0.3',
  overheadFactor: '2',
  microBatchSize: '1',
}

const GPU_IDS = GPU_PRESETS.map((gpu) => gpu.id)
const STORAGE_IDS = STORAGE_PRESETS.map((storage) => storage.id)
const MODES: InputMode[] = ['hub', 'paste', 'manual']
const DATA_MODES: DsparkDataMode[] = ['offline', 'online']

export const DSPARK_SCHEMA: UrlSchema<DsparkFormInputs> = {
  mode: { param: 'mode', default: DEFAULTS.mode, parse: enumOf(MODES) },
  provider: { param: 'provider', default: DEFAULTS.provider, parse: enumOf(['huggingface', 'modelscope']) },
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

  samples: { param: 'samples', default: DEFAULTS.samples, parse: digitsOrNull },
  sequenceLength: { param: 'seq', default: DEFAULTS.sequenceLength, parse: digitsOrNull },
  epochs: { param: 'epochs', default: DEFAULTS.epochs, parse: digitsOrNull },

  dataMode: { param: 'data', default: DEFAULTS.dataMode, parse: enumOf(DATA_MODES) },
  numTargetLayers: { param: 'target_layers', default: DEFAULTS.numTargetLayers, parse: digitsOrNull },
  numDraftLayers: { param: 'draft_layers', default: DEFAULTS.numDraftLayers, parse: digitsOrNull },
  blockSize: { param: 'block', default: DEFAULTS.blockSize, parse: digitsOrNull },
  numAnchors: { param: 'anchors', default: DEFAULTS.numAnchors, parse: digitsOrNull },
  markovRank: { param: 'markov', default: DEFAULTS.markovRank, parse: digitsOrNull },

  gpuId: { param: 'gpu', default: DEFAULTS.gpuId, parse: enumOf(GPU_IDS) },
  gpuCount: { param: 'gpus', default: DEFAULTS.gpuCount, parse: digitsOrNull },
  storageId: { param: 'storage', default: DEFAULTS.storageId, parse: enumOf(STORAGE_IDS) },

  mfu: { param: 'mfu', default: DEFAULTS.mfu, parse: decimalOrNull },
  overheadFactor: { param: 'overhead', default: DEFAULTS.overheadFactor, parse: decimalOrNull },
  microBatchSize: { param: 'micro_batch', default: DEFAULTS.microBatchSize, parse: digitsOrNull },
}

/** Fields the URL supplied, under the names the page reads. */
export interface DsparkSeeded {
  model: boolean
  gpu: boolean
  preset: boolean
}

export function useDsparkState() {
  const { values: inputs, update, serialized, seeded: fieldSeeded } =
    useUrlSyncedState<DsparkFormInputs>(DSPARK_SCHEMA)

  const seeded = useMemo(
    () => ({
      get current(): DsparkSeeded {
        const flags = fieldSeeded.current
        return {
          model: flags.modelId,
          gpu: flags.gpuId,
          preset: flags.samples || flags.sequenceLength || flags.epochs,
        }
      },
    }),
    [fieldSeeded],
  )

  return { inputs, update, serialized, seeded }
}

/** The preset whose sample count, length and epoch count match the inputs. */
export function activePresetId(
  samples: string,
  sequenceLength: string,
  epochs: string,
): string {
  const match = DSPARK_PRESETS.find(
    (preset) =>
      String(preset.samples) === samples &&
      String(preset.sequenceLength) === sequenceLength &&
      String(preset.epochs) === epochs,
  )
  return match?.id ?? 'custom'
}
