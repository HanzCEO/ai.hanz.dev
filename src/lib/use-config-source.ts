import { useMemo } from 'react'

import {
  ConfigParseError,
  MANUAL_DEFAULTS,
  manualConfig,
  parseConfigText,
  type ManualShapeInputs,
  type RawConfig,
} from '@/lib/model-config'
import { ModelConfigError, type Provider } from '@/lib/kvcache'
import { useModelConfig, type ConfigStatus, type ModelConfigState } from '@/lib/use-model-config'
import {
  defaultsFromSchema,
  digitsOrNull,
  enumOf,
  nonEmptyText,
  type UrlSchema,
} from '@/lib/url-state'

/**
 * Where a model config comes from, and the one place that resolves it.
 *
 * Three sources describe the same thing: a hub repo, a pasted config.json, and
 * a set of typed values. Every calculator that needs a model shape reads the
 * config through this module, so a page never grows its own fetch, its own
 * parser, or its own manual entry path.
 */

/** The three ways a reader can describe a model. */
export type ConfigSourceMode = 'hub' | 'paste' | 'manual'

export const DEFAULT_MODEL_ID = 'Qwen/Qwen3-8B'

/** Every field the model source needs, whatever mode is selected. */
export interface ConfigSourceInputs extends ManualShapeInputs {
  mode: ConfigSourceMode
  provider: Provider
  modelId: string
  /** A hub token. Held in memory only. */
  token: string
  /** A pasted config.json. Held in memory only, since it can be long. */
  configText: string
}

const MODES: ConfigSourceMode[] = ['hub', 'paste', 'manual']
const TIE_MODES: Array<ConfigSourceInputs['tieEmbeddings']> = ['untied', 'tied']

/**
 * The model source fields, as query string parameters.
 *
 * Declared at module scope so a page can spread this into its own schema and
 * capture one stable object in a ref. The token and the pasted config are never
 * written, because one is a secret and the other can be long.
 */
export const CONFIG_SOURCE_SCHEMA: UrlSchema<ConfigSourceInputs> = {
  mode: { param: 'mode', default: 'hub', parse: enumOf(MODES) },
  provider: {
    param: 'provider',
    default: 'huggingface',
    parse: enumOf(['huggingface', 'modelscope']),
  },
  modelId: { param: 'model', default: DEFAULT_MODEL_ID, parse: nonEmptyText },
  token: { param: 'token', default: '', parse: () => null, omit: true },
  configText: { param: 'config', default: '', parse: () => null, omit: true },

  hiddenSize: { param: 'hidden', default: MANUAL_DEFAULTS.hiddenSize, parse: digitsOrNull },
  intermediateSize: { param: 'ffn', default: MANUAL_DEFAULTS.intermediateSize, parse: digitsOrNull },
  numLayers: { param: 'layers', default: MANUAL_DEFAULTS.numLayers, parse: digitsOrNull },
  vocabSize: { param: 'vocab', default: MANUAL_DEFAULTS.vocabSize, parse: digitsOrNull },
  attentionHeads: { param: 'heads', default: MANUAL_DEFAULTS.attentionHeads, parse: digitsOrNull },
  kvHeads: { param: 'kv_heads', default: MANUAL_DEFAULTS.kvHeads, parse: digitsOrNull },
  headDim: { param: 'head_dim', default: MANUAL_DEFAULTS.headDim, parse: digitsOrNull },
  routedExperts: { param: 'experts', default: MANUAL_DEFAULTS.routedExperts, parse: digitsOrNull },
  expertsPerToken: { param: 'topk', default: MANUAL_DEFAULTS.expertsPerToken, parse: digitsOrNull },
  moeIntermediateSize: {
    param: 'expert_ffn',
    default: MANUAL_DEFAULTS.moeIntermediateSize,
    parse: digitsOrNull,
  },
  moeLayers: { param: 'moe_layers', default: MANUAL_DEFAULTS.moeLayers, parse: digitsOrNull },
  tieEmbeddings: {
    param: 'tied',
    default: MANUAL_DEFAULTS.tieEmbeddings,
    parse: enumOf(TIE_MODES),
  },
}

/** The model source values a URL supplies nothing for. */
export const CONFIG_SOURCE_DEFAULTS: ConfigSourceInputs =
  defaultsFromSchema(CONFIG_SOURCE_SCHEMA)

/**
 * The resolved config, under the same field names a hub fetch already uses.
 *
 * The names match `ModelConfigState` so a component written against a hub
 * fetch accepts this unchanged, whichever source produced it.
 */
export interface ConfigSourceState {
  status: ConfigStatus
  config: RawConfig | null
  url: string | null
  error: ModelConfigError | null
}

const IDLE: ConfigSourceState = { status: 'idle', config: null, url: null, error: null }

function errorState(message: string, kind: ModelConfigError['kind']): ConfigSourceState {
  return {
    status: 'error',
    config: null,
    url: null,
    error: new ModelConfigError(message, {
      kind,
      provider: 'huggingface',
      repo: '',
      suggestOtherProvider: false,
    }),
  }
}

/**
 * Reads a pasted config.
 *
 * Blank text is idle rather than an error, so a reader who has just switched to
 * this mode is not shown a complaint before typing anything.
 */
export function resolvePastedConfig(text: string): ConfigSourceState {
  if (text.trim() === '') return IDLE
  try {
    return { status: 'ready', config: parseConfigText(text), url: null, error: null }
  } catch (error) {
    const message =
      error instanceof ConfigParseError ? error.message : 'The config could not be read as JSON.'
    return errorState(message, 'malformed')
  }
}

/** Reads the typed values, which are written into a config shape first. */
export function resolveManualConfig(inputs: ManualShapeInputs): ConfigSourceState {
  const config = manualConfig(inputs)
  if (!config) return errorState('Enter a hidden size and a layer count of 1 or more.', 'unknown')
  return { status: 'ready', config, url: null, error: null }
}

/**
 * Resolves the model config from whichever source is selected.
 *
 * The hub hook is always called so the hook order stays stable, and its result
 * is ignored while another mode is selected. That keeps a reader who switches
 * to paste mode from losing a fetch already in flight.
 */
export function useConfigSource(inputs: ConfigSourceInputs): ConfigSourceState {
  const hub = useModelConfig(inputs.provider, inputs.modelId, inputs.token)

  const pasted = useMemo(
    () => (inputs.mode === 'paste' ? resolvePastedConfig(inputs.configText) : null),
    [inputs.mode, inputs.configText],
  )

  const manual = useMemo(
    () => (inputs.mode === 'manual' ? resolveManualConfig(inputs) : null),
    [inputs],
  )

  if (inputs.mode === 'paste') return pasted ?? IDLE
  if (inputs.mode === 'manual') return manual ?? IDLE

  if (hub.status === 'idle') return IDLE
  if (hub.status === 'loading') return { ...IDLE, status: 'loading' }
  if (hub.status === 'error') {
    return {
      ...IDLE,
      status: 'error',
      error:
        hub.error ??
        new ModelConfigError('The calculator cannot read that model config.', {
          kind: 'unknown',
          provider: inputs.provider,
          repo: inputs.modelId,
        }),
    }
  }
  if (!hub.config) return IDLE
  return { status: 'ready', config: hub.config, url: hub.url, error: null }
}

/** The hub state this hook falls back to, re-exported for a caller's props. */
export type { ModelConfigState }
