import { describe, expect, it } from 'vitest'

import type { RawConfig } from '../model-config'
import { detectModelShape } from '../model-shape'
import { loadConfigFixture } from '@/test/fixtures'

import { estimateInference } from './compute'
import {
  ACTIVATION_BYTES_PER_ELEMENT,
  ACTIVATION_FACTOR,
  DEFAULT_MTP_HEAD,
  MTP_HEADS,
  RUNTIME_OVERHEAD_BYTES,
  mtpSpeedup,
} from './presets'
import { InferenceInputError, type InferenceInputs, type MtpHeadType } from './types'

const QWEN3_8B = loadConfigFixture('qwen3-8b')

/**
 * A 70B class dense model, hand worked from the fields below.
 *
 * hidden 8192, 80 blocks, dense width 28672, vocabulary 128256, untied, 64
 * attention heads and 8 key and value heads at a head width of 128. The sum is
 * 70,552,437,504 parameters, which is the published size of this shape.
 */
const LARGE: RawConfig = {
  model_type: 'llama',
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  head_dim: 128,
  intermediate_size: 28672,
  vocab_size: 128256,
  tie_word_embeddings: false,
}

function shapeOf(config: RawConfig) {
  const shape = detectModelShape(config)
  if (!shape) throw new Error('the fixture is not a language model')
  return shape
}

function inputs(config: RawConfig, overrides: Partial<InferenceInputs> = {}): InferenceInputs {
  return {
    config,
    contextLength: 8192,
    sequences: 1,
    headroom: 0.1,
    maxGpus: 8,
    ...overrides,
  }
}

const qwen3 = () => shapeOf(QWEN3_8B)
const large = () => shapeOf(LARGE)

describe('estimateInference memory', () => {
  it('costs a plain checkpoint at two bytes for each weight', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(result.weightQuantization.mixed).toBe(false)
    expect(result.weightQuantization.primary).toBe('BF16')
    expect(result.weightsBytes).toBe(qwen3().totalParams * 2)
  })

  it('gives the same footprint for FP16 and BF16', () => {
    const bf16 = estimateInference(qwen3(), inputs(QWEN3_8B, { weightFormat: 'BF16' }))
    const fp16 = estimateInference(qwen3(), inputs(QWEN3_8B, { weightFormat: 'FP16' }))
    // The two formats differ in range, not in size, so the answer cannot move.
    expect(fp16.weightsBytes).toBe(bf16.weightsBytes)
    expect(fp16.kvCacheBytes).toBe(bf16.kvCacheBytes)
    expect(fp16.totalBytes).toBe(bf16.totalBytes)
    expect(fp16.verdict).toBe(bf16.verdict)
  })

  it('sizes the KV cache from the config', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    // 36 blocks, 8 key and value heads, a head width of 128, two tensors and
    // two bytes for each element: 36 * 2 * 8 * 128 * 2 = 147,456 bytes.
    expect(result.kvBytesPerToken).toBe(147_456)
    expect(result.kvCacheBytes).toBe(147_456 * 8192)
  })

  it('raises the KV cache with the context length', () => {
    const short = estimateInference(qwen3(), inputs(QWEN3_8B, { contextLength: 8192 }))
    const long = estimateInference(qwen3(), inputs(QWEN3_8B, { contextLength: 32768 }))
    expect(long.kvCacheBytes).toBe(short.kvCacheBytes * 4)
    expect(long.weightsBytes).toBe(short.weightsBytes)
  })

  it('raises the KV cache with the sequence count', () => {
    const one = estimateInference(qwen3(), inputs(QWEN3_8B, { sequences: 1 }))
    const eight = estimateInference(qwen3(), inputs(QWEN3_8B, { sequences: 8 }))
    expect(eight.kvCacheBytes).toBe(one.kvCacheBytes * 8)
    // The cache is per sequence, so the per token figure does not move.
    expect(eight.kvBytesPerToken).toBe(one.kvBytesPerToken)
  })

  it('sizes the activation buffer from the widest live tensor', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { sequences: 2 }))
    expect(result.activationBytes).toBe(
      2 * 8192 * 4096 * ACTIVATION_BYTES_PER_ELEMENT * ACTIVATION_FACTOR,
    )
    expect(result.runtimeReserveBytes).toBe(RUNTIME_OVERHEAD_BYTES)
  })

  it('sums the four memory terms into the total', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(result.totalBytes).toBe(
      result.weightsBytes +
        result.kvCacheBytes +
        result.activationBytes +
        result.runtimeReserveBytes,
    )
  })

  it('defaults both cache dtypes to BF16', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { weightFormat: 'MXFP4' }))
    expect(result.kvCacheDtype).toBe('BF16')
    expect(result.indexerDtype).toBe('BF16')
  })

  it('gives the same cache for an omitted dtype as for an explicit BF16', () => {
    const omitted = estimateInference(qwen3(), inputs(QWEN3_8B, { weightFormat: 'MXFP4' }))
    const explicit = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { weightFormat: 'MXFP4', kvCacheDtype: 'BF16', indexerDtype: 'BF16' }),
    )
    expect(omitted.kvCacheBytes).toBe(explicit.kvCacheBytes)
    expect(omitted.totalBytes).toBe(explicit.totalBytes)
    expect(omitted.weightsBytes).toBe(explicit.weightsBytes)
  })

  it('lowers the cache and the total for a narrower cache dtype', () => {
    const wide = estimateInference(qwen3(), inputs(QWEN3_8B, { contextLength: 32768 }))
    const narrow = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, {
        contextLength: 32768,
        kvCacheDtype: 'FP8_E4M3',
        indexerDtype: 'FP8_E4M3',
      }),
    )
    // Eight bits against sixteen, so the cache halves.
    expect(narrow.kvCacheBytes).toBe(wide.kvCacheBytes / 2)
    expect(narrow.kvBytesPerToken).toBe(wide.kvBytesPerToken / 2)
    expect(narrow.totalBytes).toBeLessThan(wide.totalBytes)
  })

  it('never lets a narrower cache dtype move the weight figure', () => {
    const wide = estimateInference(qwen3(), inputs(QWEN3_8B))
    const narrow = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { kvCacheDtype: 'FP8_E4M3', indexerDtype: 'FP8_E4M3' }),
    )
    // The weights are served in the checkpoint format whatever the cache is held in.
    expect(narrow.weightsBytes).toBe(wide.weightsBytes)
    expect(narrow.weightFormat).toBe(wide.weightFormat)
    expect(narrow.expertWeightBytes).toBe(wide.expertWeightBytes)
  })

  it('reports the cache dtypes it costed', () => {
    const result = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { kvCacheDtype: 'FP8_E4M3', indexerDtype: 'BF16' }),
    )
    expect(result.kvCacheDtype).toBe('FP8_E4M3')
    expect(result.indexerDtype).toBe('BF16')
  })

  it('lets a narrower cache dtype fit a card the wider one misses', () => {
    // A 24 GB card at a long context is the case the dtype actually decides.
    const options = { contextLength: 32768, sequences: 4, gpuFilter: ['rtx-4090'] }
    const wide = estimateInference(qwen3(), inputs(QWEN3_8B, options))
    const narrow = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, {
        ...options,
        kvCacheDtype: 'FP8_E4M3',
        indexerDtype: 'FP8_E4M3',
      }),
    )
    expect(narrow.totalBytes).toBeLessThan(wide.totalBytes)
    expect(narrow.verdict).not.toBe('none')
  })
})

describe('estimateInference verdicts', () => {
  it('reports a single card when one card holds the model', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { gpuFilter: ['rtx-4090'] }))
    expect(result.verdict).toBe('single')
    expect(result.recommended?.gpuCount).toBe(1)
    expect(result.recommended?.gpu.id).toBe('rtx-4090')
  })

  it('reports several cards when one card does not hold the model', () => {
    const result = estimateInference(large(), inputs(LARGE, { gpuFilter: ['rtx-4090'] }))
    expect(result.verdict).toBe('multi')
    expect(result.recommended?.gpuCount).toBeGreaterThan(1)
  })

  it('reports that nothing fits when the card limit is too low', () => {
    const result = estimateInference(large(), inputs(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 2 }))
    expect(result.verdict).toBe('none')
    expect(result.recommended).toBeNull()
    expect(result.alternatives).toEqual([])
    // The nearest miss is reported so the page can name the shortfall.
    expect(result.closest).not.toBeNull()
    expect(result.closest?.fits).toBe(false)
    expect(result.closest?.perCardBytes).toBeGreaterThan(result.closest?.usableBytes ?? 0)
  })

  it('reports no recommendation when the filter matches no card', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { gpuFilter: ['not-a-card'] }))
    expect(result.verdict).toBe('none')
    expect(result.recommended).toBeNull()
    expect(result.closest).toBeNull()
  })
})

describe('estimateInference ranking', () => {
  it('prefers the fewest cards', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(result.recommended?.gpuCount).toBe(1)
    for (const alternative of result.alternatives) {
      expect(alternative.gpuCount).toBeGreaterThanOrEqual(result.recommended?.gpuCount ?? 0)
    }
  })

  it('prefers the smaller card when both hold the model on one card', () => {
    const result = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['rtx-4090', 'h100'] }),
    )
    expect(result.recommended?.gpu.id).toBe('rtx-4090')
    expect(result.alternatives.map((candidate) => candidate.gpu.id)).toEqual(['h100'])
  })

  it('breaks a tie on size with the higher throughput', () => {
    // The two cards hold the same VRAM, so the faster one has to win.
    const result = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['rtx-3090', 'rtx-4090'] }),
    )
    expect(result.recommended?.gpu.vramGiB).toBe(24)
    expect(result.recommended?.gpu.id).toBe('rtx-4090')
  })

  it('raises the card count when the headroom rises', () => {
    const tight = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['rtx-5060'], headroom: 0.1 }),
    )
    const loose = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['rtx-5060'], headroom: 0.5 }),
    )
    expect(tight.recommended?.gpuCount).toBe(3)
    expect(loose.recommended?.gpuCount).toBe(8)
  })

  it('divides the weights and the cache and keeps the buffer whole', () => {
    const result = estimateInference(large(), inputs(LARGE, { gpuFilter: ['rtx-4090'] }))
    const recommended = result.recommended
    expect(recommended).not.toBeNull()
    const expected =
      (result.weightsBytes + result.kvCacheBytes) / (recommended?.gpuCount ?? 1) +
      result.activationBytes +
      result.runtimeReserveBytes
    expect(recommended?.perCardBytes).toBe(expected)
    // A second card therefore does not halve the footprint.
    expect(recommended?.perCardBytes).toBeGreaterThan(
      (result.weightsBytes + result.kvCacheBytes) / (recommended?.gpuCount ?? 1),
    )
  })
})

describe('estimateInference throughput', () => {
  it('rises with the memory bandwidth of the card', () => {
    const slow = estimateInference(qwen3(), inputs(QWEN3_8B, { gpuFilter: ['rtx-4090'] }))
    const fast = estimateInference(qwen3(), inputs(QWEN3_8B, { gpuFilter: ['h100'] }))
    expect(slow.recommended?.gpuCount).toBe(1)
    expect(fast.recommended?.gpuCount).toBe(1)
    expect(fast.decodeTokensPerSecond).toBeGreaterThan(slow.decodeTokensPerSecond)
  })

  it('rises with the card count', () => {
    // Only the headroom changes, so the bytes each token reads do not move.
    const few = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['rtx-5060'], headroom: 0.1 }),
    )
    const many = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['rtx-5060'], headroom: 0.5 }),
    )
    expect(many.recommended?.gpuCount).toBeGreaterThan(few.recommended?.gpuCount ?? 0)
    expect(many.decodeTokensPerSecond).toBeGreaterThan(few.decodeTokensPerSecond)
  })

  it('reports the figure for one sequence as well as the batch', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { sequences: 4 }))
    expect(result.perSequenceTokensPerSecond).toBeGreaterThan(0)
    expect(result.decodeTokensPerSecond).toBeGreaterThan(0)
    expect(result.decodeTokensPerSecond).toBe(result.perSequenceTokensPerSecond * 4)
  })

  it('reports no throughput when nothing fits', () => {
    const result = estimateInference(large(), inputs(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 1 }))
    expect(result.verdict).toBe('none')
    expect(result.decodeTokensPerSecond).toBe(0)
    expect(result.perSequenceTokensPerSecond).toBe(0)
  })
})

describe('estimateInference room to grow', () => {
  it('reports the free VRAM and the context it would allow', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { gpuFilter: ['h100'] }))
    expect(result.kvHeadroomBytes).toBeGreaterThan(0)
    expect(result.maxContextAtSequences).toBeGreaterThan(8192)
    expect(result.maxSequencesAtContext).toBeGreaterThan(1)
    expect(result.maxSequencesAtContext).not.toBeNull()
  })

  it('turns the free VRAM into sequences at a fixed context', () => {
    const result = estimateInference(
      qwen3(),
      inputs(QWEN3_8B, { gpuFilter: ['h100'], sequences: 4 }),
    )
    const added = (result.maxSequencesAtContext ?? 0) - 4
    const fromCache = Math.floor(
      result.kvHeadroomBytes / (result.kvBytesPerToken * 8192),
    )
    expect(added).toBe(fromCache)
  })

  it('reports no room when nothing fits', () => {
    const result = estimateInference(large(), inputs(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 1 }))
    expect(result.kvHeadroomBytes).toBe(0)
    expect(result.maxContextAtSequences).toBeNull()
    expect(result.maxSequencesAtContext).toBeNull()
  })
})

describe('estimateInference validation', () => {
  it('rejects a context length of zero', () => {
    expect(() => estimateInference(qwen3(), inputs(QWEN3_8B, { contextLength: 0 }))).toThrow(
      InferenceInputError,
    )
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { contextLength: 0 }))
    } catch (error) {
      expect((error as InferenceInputError).field).toBe('contextLength')
    }
  })

  it('rejects a sequence count of zero', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { sequences: 0 }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('sequences')
    }
  })

  it('rejects a headroom of one', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { headroom: 1 }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('headroom')
    }
  })

  it('rejects a negative headroom', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { headroom: -0.1 }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('headroom')
    }
  })

  it('rejects a maximum GPU count of zero', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { maxGpus: 0 }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('maxGpus')
    }
  })

  it('rejects an unknown weight format', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { weightFormat: 'FP4' as 'BF16' }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('weightFormat')
    }
  })
})

describe('estimateInference explanations', () => {
  it('carries steps, constants and assumptions', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(result.steps.length).toBeGreaterThan(5)
    expect(result.constants.length).toBeGreaterThan(5)
    expect(result.assumptions.length).toBeGreaterThan(5)
    for (const step of result.steps) {
      expect(step.label.trim()).not.toBe('')
      expect(step.detail.trim()).not.toBe('')
    }
  })

  it('names the weight format and the scale sidecar in the assumptions', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(
      result.assumptions.some((line) => line.includes('FP16') && line.includes('BF16')),
    ).toBe(true)
    expect(result.assumptions.some((line) => line.includes('scale sidecar'))).toBe(true)
  })

  it('states that the buffer and the reserve stay on every card', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(
      result.assumptions.some(
        (line) => line.includes('stay in full on every card') || line.includes('every card'),
      ),
    ).toBe(true)
  })

  it('carries the model notes onto the result', () => {
    // No vocab_size, no tie_word_embeddings and no intermediate_size, so the
    // shape carries a note for each. The attention fields are present, so the
    // cache engine can still size the cache.
    const thin: RawConfig = {
      model_type: 'unreleased_arch',
      hidden_size: 4096,
      num_hidden_layers: 12,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
    }
    const sparse = detectModelShape(thin)
    expect(sparse).not.toBeNull()
    const result = estimateInference(sparse!, inputs(thin))
    expect(result.notes.length).toBeGreaterThan(0)
    expect(result.bestEffort).toBe(true)
    // The cache shape had to be inferred, so the cache caveat is carried too.
    expect(result.notes.some((note) => note.includes('KV cache shape'))).toBe(true)
  })
})

describe('estimateInference on the MiMo-V2.6 releases', () => {
  /**
   * The two MiMo releases are the first fixtures with a hybrid global and
   * sliding window backbone, so they are the ones that exercise the cache split
   * and the array-valued moe_layer_freq end to end. The cache figures are the
   * ones the KV cache engine produces on its own, reached here through the
   * inference calculator.
   */
  const FLASH = loadConfigFixture('mimo-v26-flash-rl')
  const PRO = loadConfigFixture('mimo-v26-pro-rl')

  it('sizes the MiMo-V2.6-Flash-RL cache through the inference path', () => {
    const result = estimateInference(shapeOf(FLASH), inputs(FLASH, { contextLength: 32768 }))
    expect(result.kvCacheBytes).toBe(780_533_760)
    expect(result.weightQuantization.experts).toBe('MXFP4')
    expect(result.weightsBytes).toBeGreaterThan(308_778_369_024 * 0.5)
    expect(result.weightsBytes).toBeLessThan(308_778_369_024 * 0.6)
    expect(result.shape.moeLayers).toBe(47)
    expect(result.shape.attentionParams).toBe(4_482_662_400)
  })

  it('sizes the MiMo-V2.6-Pro-RL cache through the inference path', () => {
    const result = estimateInference(shapeOf(PRO), inputs(PRO, { contextLength: 32768 }))
    expect(result.kvCacheBytes).toBe(1_717_043_200)
    expect(result.weightQuantization.experts).toBe('MXFP4')
    expect(result.weightsBytes).toBeGreaterThan(1_021_247_225_856 * 0.5)
    expect(result.weightsBytes).toBeLessThan(1_021_247_225_856 * 0.6)
    expect(result.shape.moeLayers).toBe(69)
    expect(result.shape.attentionParams).toBe(18_717_081_600)
  })

  it('recommends a configuration that holds the run, or reports the shortfall', () => {
    const result = estimateInference(shapeOf(FLASH), inputs(FLASH, { contextLength: 32768 }))
    if (result.verdict === 'none') {
      expect(result.recommended).toBeNull()
      expect(result.closest).not.toBeNull()
    } else {
      expect(result.recommended).not.toBeNull()
      expect(result.recommended?.fits).toBe(true)
    }
  })
})

describe('estimateInference on a mixed checkpoint', () => {
  const V4_PRO = loadConfigFixture('deepseek-v4-pro')

  it('costs MXFP4 experts beside FP8 attention and embeddings', () => {
    const shape = shapeOf(V4_PRO)
    const result = estimateInference(shape, inputs(V4_PRO))
    expect(result.weightQuantization.mixed).toBe(true)
    expect(result.weightQuantization.experts).toBe('MXFP4')
    expect(result.weightQuantization.dense).toBe('FP8_E4M3')
    expect(result.expertWeightBytes).not.toBe(result.denseWeightBytes)
    // The mixed cost is far below the same checkpoint priced at two bytes.
    expect(result.weightsBytes).toBeLessThan(shape.totalParams * 2)
    expect(result.expertWeightBytes + result.denseWeightBytes).toBeCloseTo(result.weightsBytes, 6)
  })

  it('lets the reader force one format on every bucket', () => {
    const forced = estimateInference(shapeOf(V4_PRO), inputs(V4_PRO, { weightFormat: 'BF16' }))
    expect(forced.weightQuantization.mixed).toBe(false)
    expect(forced.weightsBytes).toBe(shapeOf(V4_PRO).totalParams * 2)
  })

  it('reads fewer active bytes from a 4 bit expert bucket', () => {
    const mixed = estimateInference(shapeOf(V4_PRO), inputs(V4_PRO))
    const bf16 = estimateInference(shapeOf(V4_PRO), inputs(V4_PRO, { weightFormat: 'BF16' }))
    expect(mixed.activeBytesPerToken).toBeGreaterThan(0)
    expect(mixed.activeBytesPerToken).toBeLessThan(bf16.activeBytesPerToken)
  })
})

describe('estimateInference MTP heads', () => {
  /** Every head except none, which is the baseline and not a head at all. */
  const HEADS: MtpHeadType[] = ['sequential-mtp', 'parallel-mtp', 'medusa', 'eagle-3']

  it('offers five heads, from none to EAGLE-3', () => {
    expect(MTP_HEADS.map((head) => head.id)).toEqual([
      'none',
      'sequential-mtp',
      'parallel-mtp',
      'medusa',
      'eagle-3',
    ])
    expect(MTP_HEADS.map((head) => head.speedup)).toEqual([1, 1.5, 1.4, 1.6, 2])
    for (const head of MTP_HEADS) {
      // Every row states the figure it was held to and where that figure came
      // from, so a reader can check the number rather than trust it.
      expect(head.published.trim()).not.toBe('')
      expect(head.source.trim()).not.toBe('')
      expect(head.hint.trim()).not.toBe('')
    }
  })

  it('holds every applied multiplier at or below the published figure', () => {
    for (const head of MTP_HEADS) {
      // The published figure is a sentence, so the first number in it is the
      // one the multiplier is held below.
      const published = Number(/(\d+(?:\.\d+)?)/.exec(head.published)?.[1])
      expect(Number.isFinite(published), head.id).toBe(true)
      expect(head.speedup, head.id).toBeLessThanOrEqual(published)
    }
  })

  it('treats an absent head as no head at all', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(result.mtpHead).toBe(DEFAULT_MTP_HEAD)
    expect(result.mtpHead).toBe('none')
    expect(result.mtpSpeedup).toBe(1)
    expect(result.decodeTokensPerSecond).toBe(result.baseDecodeTokensPerSecond)
    expect(result.perSequenceTokensPerSecond).toBe(result.basePerSequenceTokensPerSecond)
  })

  it('treats an explicit none as the same answer as an absent head', () => {
    const absent = estimateInference(qwen3(), inputs(QWEN3_8B))
    const explicit = estimateInference(qwen3(), inputs(QWEN3_8B, { mtpHead: 'none' }))
    expect(explicit.decodeTokensPerSecond).toBe(absent.decodeTokensPerSecond)
    expect(explicit.baseDecodeTokensPerSecond).toBe(absent.baseDecodeTokensPerSecond)
  })

  it('raises both rates by the multiplier of the head', () => {
    const baseline = estimateInference(qwen3(), inputs(QWEN3_8B))
    for (const head of HEADS) {
      const result = estimateInference(qwen3(), inputs(QWEN3_8B, { mtpHead: head }))
      const speedup = mtpSpeedup(head)
      expect(result.mtpHead, head).toBe(head)
      expect(result.mtpSpeedup, head).toBe(speedup)
      // The roofline is the same figure whichever head is selected, because the
      // head is a property of the checkpoint and not of the card.
      expect(result.baseDecodeTokensPerSecond, head).toBeCloseTo(
        baseline.decodeTokensPerSecond,
        6,
      )
      expect(result.decodeTokensPerSecond, head).toBeCloseTo(
        result.baseDecodeTokensPerSecond * speedup,
        6,
      )
      expect(result.perSequenceTokensPerSecond, head).toBeCloseTo(
        result.basePerSequenceTokensPerSecond * speedup,
        6,
      )
      expect(result.decodeTokensPerSecond, head).toBeGreaterThan(baseline.decodeTokensPerSecond)
    }
  })

  it('keeps the ratio of the batch rate to the single sequence rate', () => {
    for (const head of HEADS) {
      const result = estimateInference(qwen3(), inputs(QWEN3_8B, { sequences: 4, mtpHead: head }))
      expect(result.decodeTokensPerSecond, head).toBeCloseTo(
        result.perSequenceTokensPerSecond * 4,
        6,
      )
    }
  })

  it('never moves a memory term, a verdict or a recommendation', () => {
    const baseline = estimateInference(qwen3(), inputs(QWEN3_8B, { gpuFilter: ['rtx-4090'] }))
    for (const head of HEADS) {
      const result = estimateInference(
        qwen3(),
        inputs(QWEN3_8B, { gpuFilter: ['rtx-4090'], mtpHead: head }),
      )
      // The head adds a small number of weights against the whole checkpoint,
      // so the hardware answer cannot change with it.
      expect(result.weightsBytes, head).toBe(baseline.weightsBytes)
      expect(result.kvCacheBytes, head).toBe(baseline.kvCacheBytes)
      expect(result.activationBytes, head).toBe(baseline.activationBytes)
      expect(result.runtimeReserveBytes, head).toBe(baseline.runtimeReserveBytes)
      expect(result.totalBytes, head).toBe(baseline.totalBytes)
      expect(result.verdict, head).toBe(baseline.verdict)
      expect(result.recommended?.gpu.id, head).toBe(baseline.recommended?.gpu.id)
      expect(result.recommended?.gpuCount, head).toBe(baseline.recommended?.gpuCount)
      expect(result.recommended?.perCardBytes, head).toBe(baseline.recommended?.perCardBytes)
      expect(result.maxContextAtSequences, head).toBe(baseline.maxContextAtSequences)
    }
  })

  it('reports no throughput for any head when nothing fits', () => {
    for (const head of HEADS) {
      const result = estimateInference(
        large(),
        inputs(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 1, mtpHead: head }),
      )
      expect(result.verdict, head).toBe('none')
      expect(result.decodeTokensPerSecond, head).toBe(0)
      expect(result.baseDecodeTokensPerSecond, head).toBe(0)
      expect(result.perSequenceTokensPerSecond, head).toBe(0)
      expect(result.basePerSequenceTokensPerSecond, head).toBe(0)
    }
  })

  it('rejects a head it does not know', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { mtpHead: 'medusa-2' as MtpHeadType }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('mtpHead')
      expect((error as InferenceInputError).message).toContain('medusa-2')
    }
  })

  it('names the head in a step, a constant and an assumption', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B, { mtpHead: 'eagle-3' }))
    const step = result.steps.find((entry) => entry.label === 'MTP head')
    expect(step).toBeDefined()
    expect(step?.detail).toContain('EAGLE-3 head')
    expect(step?.detail).toContain('2x')
    expect(step?.detail).toContain('estimate for measurement')

    expect(result.constants.some((entry) => entry.key === 'mtp_head')).toBe(true)
    expect(result.constants.some((entry) => entry.key === 'mtp_speedup')).toBe(true)
    expect(
      result.constants.find((entry) => entry.key === 'mtp_speedup')?.value,
    ).toBe(2)

    expect(
      result.assumptions.some(
        (line) => line.includes('does not transfer to another') && line.includes('estimate'),
      ),
    ).toBe(true)
  })

  it('says the answer stays on the roofline when no head is selected', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    const step = result.steps.find((entry) => entry.label === 'MTP head')
    expect(step).toBeDefined()
    expect(step?.detail).toContain('roofline')
    expect(step?.detail).toContain('estimate for measurement')
  })
})
