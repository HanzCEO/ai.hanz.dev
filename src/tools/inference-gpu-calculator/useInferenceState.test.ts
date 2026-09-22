import { describe, expect, it } from 'vitest'

import { defaultsFromSchema, serializeState } from '@/lib/url-state'
import {
  CALCULATOR_SCHEMA,
  type CalculatorInputs,
} from '@/tools/kv-cache-calculator/useCalculatorState'

import { INFERENCE_SCHEMA, type InferenceFormInputs } from './useInferenceState'

const DEFAULTS = defaultsFromSchema(INFERENCE_SCHEMA)

describe('INFERENCE_SCHEMA', () => {
  it('covers every field the page reads', () => {
    const expected: Array<keyof InferenceFormInputs> = [
      // The model source, shared with the cache layer.
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
      // The cache layer's own fields.
      'contextLength',
      'sequenceCount',
      'kvCacheDtype',
      'indexerDtype',
      // The hardware fields this layer adds.
      'precision',
      'mtpHead',
      'headroomPercent',
      'maxGpus',
    ]
    expect(Object.keys(INFERENCE_SCHEMA).sort()).toEqual([...expected].sort())
  })

  it('extends the cache schema rather than restating it', () => {
    const cacheKeys = Object.keys(CALCULATOR_SCHEMA)
    for (const key of cacheKeys) {
      expect(INFERENCE_SCHEMA[key as keyof InferenceFormInputs]).toBe(
        CALCULATOR_SCHEMA[key as keyof CalculatorInputs],
      )
    }
  })

  it('adds only the four hardware fields', () => {
    const added = Object.keys(INFERENCE_SCHEMA).filter(
      (key) => !(key in CALCULATOR_SCHEMA),
    )
    expect(added.sort()).toEqual(['headroomPercent', 'maxGpus', 'mtpHead', 'precision'])
  })

  it('opens on BF16, no MTP head, a 10 percent headroom, and a limit of 8 cards', () => {
    expect(DEFAULTS.precision).toBe('BF16')
    expect(DEFAULTS.mtpHead).toBe('none')
    expect(DEFAULTS.headroomPercent).toBe('10')
    expect(DEFAULTS.maxGpus).toBe('8')
  })

  it('writes nothing into the query string at the defaults', () => {
    expect(serializeState(INFERENCE_SCHEMA, DEFAULTS)).toBe('')
  })

  it('never writes the token or the pasted config', () => {
    const query = serializeState(INFERENCE_SCHEMA, {
      ...DEFAULTS,
      token: 'hf_a_secret_token',
      configText: '{"model_type":"qwen3"}',
    })
    // The token is a secret and a config can be long, so neither belongs in a
    // URL that a reader might share.
    expect(query).not.toContain('token')
    expect(query).not.toContain('hf_a_secret_token')
    expect(query).not.toContain('config=')
  })

  it('round-trips every field through the query string', () => {
    const values: InferenceFormInputs = {
      ...DEFAULTS,
      mode: 'manual',
      provider: 'modelscope',
      modelId: 'openbmb/MiniCPM5-2B',
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
      contextLength: '65536',
      sequenceCount: '8',
      kvCacheDtype: 'FP8_E4M3',
      indexerDtype: 'FP4',
      precision: 'FP16',
      mtpHead: 'eagle-3',
      headroomPercent: '12.5',
      maxGpus: '4',
    }

    const params = new URLSearchParams(serializeState(INFERENCE_SCHEMA, values))

    for (const key of Object.keys(INFERENCE_SCHEMA) as Array<keyof InferenceFormInputs>) {
      const spec = INFERENCE_SCHEMA[key]
      if (spec.omit) continue
      const raw = params.get(spec.param)
      expect(raw, `${String(key)} was not written`).not.toBeNull()
      expect(spec.parse(raw), `${String(key)} did not parse back`).toEqual(values[key])
    }
  })

  it('falls back to the default when the URL names an unknown precision', () => {
    // A parser returns null for anything outside the allowed set, and the hook
    // then keeps the field at its default rather than accepting the value.
    expect(INFERENCE_SCHEMA.precision.parse('FP8')).toBeNull()
    expect(INFERENCE_SCHEMA.precision.default).toBe('BF16')
  })

  it('rejects an unknown mode and an unknown tie setting', () => {
    expect(INFERENCE_SCHEMA.mode.parse('remote')).toBeNull()
    expect(INFERENCE_SCHEMA.tieEmbeddings.parse('shared')).toBeNull()
  })

  it('accepts the two weight precisions and no others', () => {
    expect(INFERENCE_SCHEMA.precision.parse('FP16')).toBe('FP16')
    expect(INFERENCE_SCHEMA.precision.parse('BF16')).toBe('BF16')
    expect(INFERENCE_SCHEMA.precision.parse('INT8')).toBeNull()
  })

  it('accepts the five MTP heads and no others', () => {
    expect(INFERENCE_SCHEMA.mtpHead.parse('none')).toBe('none')
    expect(INFERENCE_SCHEMA.mtpHead.parse('sequential-mtp')).toBe('sequential-mtp')
    expect(INFERENCE_SCHEMA.mtpHead.parse('parallel-mtp')).toBe('parallel-mtp')
    expect(INFERENCE_SCHEMA.mtpHead.parse('medusa')).toBe('medusa')
    expect(INFERENCE_SCHEMA.mtpHead.parse('eagle-3')).toBe('eagle-3')
    // A head that only exists in someone else's fork is not offered here.
    expect(INFERENCE_SCHEMA.mtpHead.parse('medusa-2')).toBeNull()
    expect(INFERENCE_SCHEMA.mtpHead.parse('')).toBeNull()
  })

  it('writes the MTP head into the query string under mtp', () => {
    expect(INFERENCE_SCHEMA.mtpHead.param).toBe('mtp')
    // Declared with the input type so the literal is not widened to string,
    // which would make the schema argument fail to type check.
    const medusa: InferenceFormInputs = { ...DEFAULTS, mtpHead: 'medusa' }
    const query = serializeState(INFERENCE_SCHEMA, medusa)
    expect(new URLSearchParams(query).get('mtp')).toBe('medusa')
    // The default head is the baseline, so it writes nothing at all.
    const baseline: InferenceFormInputs = { ...DEFAULTS, mtpHead: 'none' }
    expect(serializeState(INFERENCE_SCHEMA, baseline)).toBe('')
  })

  it('accepts the cache dtypes the cache layer offers', () => {
    expect(INFERENCE_SCHEMA.kvCacheDtype.parse('FP8_E4M3')).toBe('FP8_E4M3')
    expect(INFERENCE_SCHEMA.kvCacheDtype.parse('BF16')).toBe('BF16')
    expect(INFERENCE_SCHEMA.kvCacheDtype.parse('NOPE')).toBeNull()
  })

  it('rejects a fractional context length and a negative headroom', () => {
    expect(INFERENCE_SCHEMA.contextLength.parse('8192.5')).toBeNull()
    expect(INFERENCE_SCHEMA.headroomPercent.parse('-5')).toBeNull()
    expect(INFERENCE_SCHEMA.headroomPercent.parse('12.5')).toBe('12.5')
  })
})
