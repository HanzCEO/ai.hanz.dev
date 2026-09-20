import { describe, expect, it } from 'vitest'

import { MANUAL_DEFAULTS } from '@/lib/model-config'
import { defaultsFromSchema, serializeState } from '@/lib/url-state'
import { readConfigFixtureText } from '@/test/fixtures'

import {
  CONFIG_SOURCE_DEFAULTS,
  CONFIG_SOURCE_SCHEMA,
  resolveManualConfig,
  resolvePastedConfig,
  type ConfigSourceInputs,
} from './use-config-source'

describe('CONFIG_SOURCE_SCHEMA', () => {
  it('covers every field of the model source', () => {
    const expected: Array<keyof ConfigSourceInputs> = [
      'mode',
      'provider',
      'modelId',
      'token',
      'configText',
      'hiddenSize',
      'intermediateSize',
      'numLayers',
      'vocabSize',
      'attentionHeads',
      'kvHeads',
      'headDim',
      'routedExperts',
      'expertsPerToken',
      'moeIntermediateSize',
      'moeLayers',
      'tieEmbeddings',
    ]
    expect(Object.keys(CONFIG_SOURCE_SCHEMA).sort()).toEqual([...expected].sort())
  })

  it('opens on a hub model in the first provider', () => {
    expect(CONFIG_SOURCE_DEFAULTS.mode).toBe('hub')
    expect(CONFIG_SOURCE_DEFAULTS.provider).toBe('huggingface')
    expect(CONFIG_SOURCE_DEFAULTS.modelId).toBe('Qwen/Qwen3-8B')
  })

  it('writes nothing into the query string at the defaults', () => {
    expect(serializeState(CONFIG_SOURCE_SCHEMA, CONFIG_SOURCE_DEFAULTS)).toBe('')
  })

  it('agrees with defaultsFromSchema', () => {
    expect(defaultsFromSchema(CONFIG_SOURCE_SCHEMA)).toEqual(CONFIG_SOURCE_DEFAULTS)
  })

  it('never writes the token or the pasted config', () => {
    const query = serializeState(CONFIG_SOURCE_SCHEMA, {
      ...CONFIG_SOURCE_DEFAULTS,
      token: 'hf_a_secret_token',
      configText: '{"model_type":"qwen3"}',
    })
    expect(query).not.toContain('token')
    expect(query).not.toContain('hf_a_secret_token')
    expect(query).not.toContain('config=')
  })

  it('round-trips every field through the query string', () => {
    const values: ConfigSourceInputs = {
      mode: 'manual',
      provider: 'modelscope',
      modelId: 'openbmb/MiniCPM5-2B',
      token: '',
      configText: '',
      hiddenSize: '2048',
      intermediateSize: '6144',
      numLayers: '42',
      vocabSize: '130560',
      attentionHeads: '16',
      kvHeads: '2',
      headDim: '64',
      routedExperts: '64',
      expertsPerToken: '6',
      moeIntermediateSize: '768',
      moeLayers: '21',
      tieEmbeddings: 'tied',
    }

    const params = new URLSearchParams(serializeState(CONFIG_SOURCE_SCHEMA, values))

    for (const key of Object.keys(CONFIG_SOURCE_SCHEMA) as Array<keyof ConfigSourceInputs>) {
      const spec = CONFIG_SOURCE_SCHEMA[key]
      if (spec.omit) continue
      const raw = params.get(spec.param)
      expect(raw, `${String(key)} was not written`).not.toBeNull()
      expect(spec.parse(raw), `${String(key)} did not parse back`).toEqual(values[key])
    }
  })

  it('rejects a mode and a tie setting outside the offered set', () => {
    expect(CONFIG_SOURCE_SCHEMA.mode.parse('remote')).toBeNull()
    expect(CONFIG_SOURCE_SCHEMA.tieEmbeddings.parse('shared')).toBeNull()
    expect(CONFIG_SOURCE_SCHEMA.mode.parse('paste')).toBe('paste')
  })
})

describe('resolvePastedConfig', () => {
  it('is idle while the field is empty', () => {
    expect(resolvePastedConfig('').status).toBe('idle')
    expect(resolvePastedConfig('   \n ').status).toBe('idle')
  })

  it('reads a real config fixture', () => {
    const state = resolvePastedConfig(readConfigFixtureText('qwen3-8b'))
    expect(state.status).toBe('ready')
    expect(state.error).toBeNull()
    expect(state.url).toBeNull()
    expect(state.config?.model_type).toBe('qwen3')
    expect(state.config?.num_hidden_layers).toBe(36)
  })

  it('reports text that is not JSON as a malformed config', () => {
    const state = resolvePastedConfig('not json')
    expect(state.status).toBe('error')
    expect(state.config).toBeNull()
    expect(state.error?.kind).toBe('malformed')
    expect(state.error?.suggestOtherProvider).toBe(false)
    expect(state.error?.message).not.toBe('')
  })

  it('reports a JSON value that is not a config object', () => {
    const state = resolvePastedConfig('[1, 2, 3]')
    expect(state.status).toBe('error')
    expect(state.error?.kind).toBe('malformed')
  })
})

describe('resolveManualConfig', () => {
  it('reads the defaults as a dense model', () => {
    const state = resolveManualConfig(MANUAL_DEFAULTS)
    expect(state.status).toBe('ready')
    expect(state.error).toBeNull()
    expect(state.config?.hidden_size).toBe(4096)
    expect(state.config?.num_hidden_layers).toBe(36)
  })

  it('reports a missing hidden size as an error', () => {
    const state = resolveManualConfig({ ...MANUAL_DEFAULTS, hiddenSize: '' })
    expect(state.status).toBe('error')
    expect(state.config).toBeNull()
    expect(state.error?.kind).toBe('unknown')
    expect(state.error?.message).toBe('Enter a hidden size and a layer count of 1 or more.')
  })

  it('reports a missing layer count as an error', () => {
    const state = resolveManualConfig({ ...MANUAL_DEFAULTS, numLayers: '0' })
    expect(state.status).toBe('error')
    expect(state.error?.kind).toBe('unknown')
  })
})
