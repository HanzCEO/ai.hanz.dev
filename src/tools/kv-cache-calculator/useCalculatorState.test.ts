import { describe, expect, it } from 'vitest'

import { defaultsFromSchema, parsePositiveInteger, serializeState } from '@/lib/url-state'
import {
  CONFIG_SOURCE_DEFAULTS,
  CONFIG_SOURCE_SCHEMA,
  type ConfigSourceInputs,
} from '@/lib/use-config-source'

import {
  CALCULATOR_SCHEMA,
  DEFAULT_CONTEXT_LENGTH,
  DEFAULT_MODEL_ID,
  DEFAULT_SEQUENCE_COUNT,
  type CalculatorInputs,
} from './useCalculatorState'

/**
 * These guard the shareable link contract of the KV cache calculator: a page
 * opened with no query string stays clean, and every field survives a trip
 * through the URL.
 */
describe('CALCULATOR_SCHEMA', () => {
  const defaults = defaultsFromSchema(CALCULATOR_SCHEMA)

  it('produces an empty query string at every default', () => {
    expect(serializeState(CALCULATOR_SCHEMA, defaults)).toBe('')
  })

  it('starts on the documented defaults', () => {
    expect(defaults.modelId).toBe(DEFAULT_MODEL_ID)
    expect(defaults.contextLength).toBe(DEFAULT_CONTEXT_LENGTH)
    expect(defaults.sequenceCount).toBe(DEFAULT_SEQUENCE_COUNT)
    expect(defaults.provider).toBe('huggingface')
    expect(defaults.kvCacheDtype).toBe('BF16')
    expect(defaults.indexerDtype).toBe('BF16')
  })

  it('round-trips each field the calculator writes', () => {
    const overrides: Array<[string, string]> = [
      ['provider', 'modelscope'],
      ['model', 'zai-org/GLM-5.3'],
      ['context', '131072'],
      ['sequences', '8'],
      ['kv_dtype', 'FP8_E4M3'],
      ['indexer_dtype', 'FP4'],
    ]

    for (const [param, value] of overrides) {
      const values = { ...defaults, ...valuesFor(param, value) }
      const query = new URLSearchParams(serializeState(CALCULATOR_SCHEMA, values))
      expect(query.get(param)).toBe(value)
    }
  })

  it('never writes the token, which is a secret', () => {
    const values = { ...defaults, token: 'hf_abcdefghijklmnop' }
    const query = serializeState(CALCULATOR_SCHEMA, values)
    expect(query).toBe('')
    expect(query).not.toContain('hf_')
  })

  it('reads a numeric string back through its parse function', () => {
    expect(CALCULATOR_SCHEMA.contextLength.parse('131072')).toBe('131072')
    expect(CALCULATOR_SCHEMA.contextLength.parse('abc')).toBeNull()
    expect(CALCULATOR_SCHEMA.modelId.parse('  Qwen/Qwen3-8B ')).toBe('Qwen/Qwen3-8B')
    expect(CALCULATOR_SCHEMA.modelId.parse('   ')).toBeNull()
    expect(CALCULATOR_SCHEMA.kvCacheDtype.parse('FP4')).toBe('FP4')
    expect(CALCULATOR_SCHEMA.kvCacheDtype.parse('NOPE')).toBeNull()
    expect(CALCULATOR_SCHEMA.provider.parse('modelscope')).toBe('modelscope')
    expect(CALCULATOR_SCHEMA.provider.parse('github')).toBeNull()
  })

  it('covers every field of the page', () => {
    const expected: Array<keyof CalculatorInputs> = [
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
      'contextLength',
      'sequenceCount',
      'kvCacheDtype',
      'indexerDtype',
    ]
    expect(Object.keys(CALCULATOR_SCHEMA).sort()).toEqual([...expected].sort())
  })

  it('takes the model source fields from the shared schema', () => {
    for (const key of Object.keys(CONFIG_SOURCE_SCHEMA)) {
      expect(CALCULATOR_SCHEMA[key as keyof CalculatorInputs]).toBe(
        CONFIG_SOURCE_SCHEMA[key as keyof ConfigSourceInputs],
      )
    }
    expect(CALCULATOR_SCHEMA.modelId.default).toBe(CONFIG_SOURCE_DEFAULTS.modelId)
  })

  it('adds only the four cache fields', () => {
    const added = Object.keys(CALCULATOR_SCHEMA).filter((key) => !(key in CONFIG_SOURCE_SCHEMA))
    expect(added.sort()).toEqual([
      'contextLength',
      'indexerDtype',
      'kvCacheDtype',
      'sequenceCount',
    ])
  })

  it('round-trips the model source fields it shares with the inference page', () => {
    const values: CalculatorInputs = {
      ...defaults,
      mode: 'manual',
      hiddenSize: '2048',
      numLayers: '42',
      kvHeads: '2',
      tieEmbeddings: 'tied',
    }
    const params = new URLSearchParams(serializeState(CALCULATOR_SCHEMA, values))
    expect(params.get('mode')).toBe('manual')
    expect(params.get('hidden')).toBe('2048')
    expect(params.get('layers')).toBe('42')
    expect(params.get('kv_heads')).toBe('2')
    expect(params.get('tied')).toBe('tied')
  })
})

/** Maps a query parameter name back onto the field it belongs to. */
function valuesFor(param: string, value: string): Record<string, string> {
  switch (param) {
    case 'provider':
      return { provider: value }
    case 'model':
      return { modelId: value }
    case 'context':
      return { contextLength: value }
    case 'sequences':
      return { sequenceCount: value }
    case 'kv_dtype':
      return { kvCacheDtype: value }
    case 'indexer_dtype':
      return { indexerDtype: value }
    default:
      throw new Error(`Unmapped parameter: ${param}`)
  }
}

describe('parsePositiveInteger', () => {
  it('accepts a whole number of one or more', () => {
    expect(parsePositiveInteger('1')).toBe(1)
    expect(parsePositiveInteger(' 32768 ')).toBe(32768)
  })

  it('rejects zero, a negative, a fraction, and text', () => {
    expect(parsePositiveInteger('0')).toBeNull()
    expect(parsePositiveInteger('-5')).toBeNull()
    expect(parsePositiveInteger('1.5')).toBeNull()
    expect(parsePositiveInteger('many')).toBeNull()
    expect(parsePositiveInteger('')).toBeNull()
  })
})
