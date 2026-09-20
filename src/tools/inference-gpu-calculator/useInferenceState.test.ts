import { describe, expect, it } from 'vitest'

import { defaultsFromSchema, serializeState } from '@/lib/url-state'

import { INFERENCE_SCHEMA, type InferenceFormInputs } from './useInferenceState'

const DEFAULTS = defaultsFromSchema(INFERENCE_SCHEMA)

describe('INFERENCE_SCHEMA', () => {
  it('covers every field of the form', () => {
    const expected: Array<keyof InferenceFormInputs> = [
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
      'precision',
      'contextLength',
      'sequences',
      'headroomPercent',
      'maxGpus',
    ]
    expect(Object.keys(INFERENCE_SCHEMA).sort()).toEqual([...expected].sort())
  })

  it('opens on a hub model in BF16 with a single sequence', () => {
    expect(DEFAULTS.mode).toBe('hub')
    expect(DEFAULTS.modelId).toBe('Qwen/Qwen3-8B')
    expect(DEFAULTS.precision).toBe('BF16')
    expect(DEFAULTS.contextLength).toBe('8192')
    expect(DEFAULTS.sequences).toBe('1')
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
    expect(query).not.toContain('config')
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
      precision: 'FP16',
      contextLength: '32768',
      sequences: '8',
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

  it('accepts the two precisions and no others', () => {
    expect(INFERENCE_SCHEMA.precision.parse('FP16')).toBe('FP16')
    expect(INFERENCE_SCHEMA.precision.parse('BF16')).toBe('BF16')
    expect(INFERENCE_SCHEMA.precision.parse('INT8')).toBeNull()
  })

  it('rejects a fractional context length and a negative headroom', () => {
    expect(INFERENCE_SCHEMA.contextLength.parse('8192.5')).toBeNull()
    expect(INFERENCE_SCHEMA.headroomPercent.parse('-5')).toBeNull()
    expect(INFERENCE_SCHEMA.headroomPercent.parse('12.5')).toBe('12.5')
  })
})
