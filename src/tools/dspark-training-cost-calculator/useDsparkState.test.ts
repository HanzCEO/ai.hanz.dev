import { describe, expect, it } from 'vitest'

import { detectDsparkShape } from '@/lib/dspark'
import { defaultsFromSchema, serializeState } from '@/lib/url-state'

import { DSPARK_SCHEMA, activePresetId, type DsparkFormInputs } from './useDsparkState'
import { manualConfig } from './useDsparkShape'

/**
 * These guard the shareable link contract and the manual entry path.
 *
 * The manual path is what the page opens on, so a break in it is not a corner
 * case: it is the default view, and the config it builds has to survive the same
 * detector every other mode uses.
 */
/**
 * A complete input set, typed as the form's own shape.
 *
 * The annotation matters: an inline object literal widens the enum fields to
 * string, and the schema is not assignable from that wider form.
 */
function values(overrides: Partial<DsparkFormInputs> = {}): DsparkFormInputs {
  return { ...(defaultsFromSchema(DSPARK_SCHEMA) as DsparkFormInputs), ...overrides }
}

describe('DSPARK_SCHEMA', () => {
  const defaults = values()

  it('opens on manual entry with the MiniCPM5-2B-DSpark recipe', () => {
    expect(defaults.mode).toBe('manual')
    expect(defaults.dataMode).toBe('offline')
    // The recipe OpenBMB published: 1,959,525 sequences read six times.
    expect(defaults.samples).toBe('1959525')
    expect(defaults.sequenceLength).toBe('3600')
    expect(defaults.epochs).toBe('6')
    // Anchors are not published, so they open at one block per sequence token.
    expect(defaults.numAnchors).toBe('514')
  })

  it('opens on MiniCPM5-2B as the target', () => {
    expect(defaults.hiddenSize).toBe('2048')
    expect(defaults.numLayers).toBe('42')
    expect(defaults.intermediateSize).toBe('6144')
    expect(defaults.vocabSize).toBe('130560')
    expect(defaults.attentionHeads).toBe('16')
    expect(defaults.kvHeads).toBe('2')
    expect(defaults.headDim).toBe('128')
  })

  it('matches the draft recipe both published checkpoints use', () => {
    expect(defaults.blockSize).toBe('7')
    expect(defaults.numDraftLayers).toBe('5')
    expect(defaults.numTargetLayers).toBe('5')
    expect(defaults.markovRank).toBe('256')
  })

  it('opens the target on the automatic weight format', () => {
    expect(defaults.targetWeightFormat).toBe('auto')
  })

  it('writes nothing to the URL when every field is at its default', () => {
    expect(serializeState(DSPARK_SCHEMA, defaults)).toBe('')
  })

  it('keeps the token and the pasted config out of the URL', () => {
    const serialized = serializeState(
      DSPARK_SCHEMA,
      values({ token: 'hf_secret', configText: '{"model_type":"qwen3"}' }),
    )
    expect(serialized).not.toContain('hf_secret')
    expect(serialized).not.toContain('model_type')
  })

  it('round trips a changed value', () => {
    const serialized = serializeState(
      DSPARK_SCHEMA,
      values({ dataMode: 'online', numTargetLayers: '2', gpuCount: '4', targetWeightFormat: 'MXFP4' }),
    )
    const params = new URLSearchParams(serialized)
    expect(params.get('data')).toBe('online')
    expect(params.get('target_layers')).toBe('2')
    expect(params.get('gpus')).toBe('4')
    expect(params.get('target_weights')).toBe('MXFP4')
  })

  it('rejects a data mode it does not recognise', () => {
    expect(DSPARK_SCHEMA.dataMode.parse('sideways')).toBeNull()
    expect(DSPARK_SCHEMA.dataMode.parse('online')).toBe('online')
  })

  it('accepts the automatic weight format and every format id', () => {
    expect(DSPARK_SCHEMA.targetWeightFormat.parse('auto')).toBe('auto')
    expect(DSPARK_SCHEMA.targetWeightFormat.parse('BF16')).toBe('BF16')
    expect(DSPARK_SCHEMA.targetWeightFormat.parse('FP8_E4M3')).toBe('FP8_E4M3')
    expect(DSPARK_SCHEMA.targetWeightFormat.parse('MXFP4')).toBe('MXFP4')
    expect(DSPARK_SCHEMA.targetWeightFormat.parse('NVFP4')).toBe('NVFP4')
    expect(DSPARK_SCHEMA.targetWeightFormat.parse('FP4')).toBeNull()
  })
})

describe('activePresetId', () => {
  it('recognises the MiniCPM5-2B-DSpark recipe', () => {
    expect(activePresetId('1959525', '3600', '6')).toBe('minicpm5-2b-dspark')
  })

  it('recognises the DeepSpec paper recipe', () => {
    expect(activePresetId('302000', '4096', '10')).toBe('deepspec')
  })

  it('falls back to custom when any field differs', () => {
    expect(activePresetId('302000', '4096', '9')).toBe('custom')
    expect(activePresetId('1', '4096', '10')).toBe('custom')
  })
})

describe('manualConfig', () => {
  it('builds a dense target from the defaults', () => {
    const config = manualConfig(values())
    expect(config).not.toBeNull()
    expect(config?.n_routed_experts).toBeUndefined()

    const shape = detectDsparkShape(config!)
    expect(shape).not.toBeNull()
    expect(shape?.hiddenSize).toBe(2048)
    expect(shape?.numLayers).toBe(42)
    expect(shape?.moeLayers).toBe(0)
    expect(shape?.looksLikeDraftConfig).toBe(false)
    // MiniCPM5-2B is untied, so the vocabulary is stored twice. The published
    // checkpoint holds 2,516,756,480 parameters.
    expect(shape?.totalParams).toBe(2_516_582_400)
  })

  it('builds an expert bank when a routed expert count is given', () => {
    const config = manualConfig(
      values({
        hiddenSize: '2048',
        intermediateSize: '6144',
        numLayers: '48',
        routedExperts: '128',
        expertsPerToken: '8',
        moeIntermediateSize: '768',
        moeLayers: '48',
      }),
    )
    const shape = detectDsparkShape(config!)
    expect(shape?.routedExperts).toBe(128)
    expect(shape?.moeLayers).toBe(48)
    // Every block carries an expert bank, so the dense feed forward width is
    // never used and the target is entirely sparse.
    expect(shape?.activeParamsPerToken).toBeLessThan(shape!.totalParams / 4)
  })

  it('records the dense block count, which is how the detector reads a partial bank', () => {
    const config = manualConfig(values({ numLayers: '48', routedExperts: '128', moeLayers: '24' }))
    expect(config?.first_k_dense_replace).toBe(24)
    expect(detectDsparkShape(config!)?.moeLayers).toBe(24)
  })

  it('returns null when the shape is not described at all', () => {
    expect(manualConfig(values({ hiddenSize: '', numLayers: '' }))).toBeNull()
  })

  it('ignores an expert count of zero rather than building an empty bank', () => {
    const config = manualConfig(values({ routedExperts: '0', moeLayers: '0' }))
    expect(config?.n_routed_experts).toBeUndefined()
  })
})
