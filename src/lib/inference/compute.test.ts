import { describe, expect, it } from 'vitest'

import type { RawConfig } from '../model-config'
import { detectModelShape } from '../model-shape'
import { loadConfigFixture } from '@/test/fixtures'

import { estimateInference } from './compute'
import {
  ACTIVATION_BYTES_PER_ELEMENT,
  ACTIVATION_FACTOR,
  BYTES_PER_WEIGHT,
  RUNTIME_OVERHEAD_BYTES,
} from './presets'
import { InferenceInputError, type InferenceInputs } from './types'

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
    precision: 'BF16',
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
  it('counts two bytes for each weight', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(result.bytesPerWeight).toBe(2)
    expect(result.weightsBytes).toBe(qwen3().totalParams * BYTES_PER_WEIGHT)
  })

  it('gives the same footprint for FP16 and BF16', () => {
    const bf16 = estimateInference(qwen3(), inputs(QWEN3_8B, { precision: 'BF16' }))
    const fp16 = estimateInference(qwen3(), inputs(QWEN3_8B, { precision: 'FP16' }))
    // The two precisions differ in range, not in size, so the answer cannot move.
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

  it('rejects an unknown precision', () => {
    try {
      estimateInference(qwen3(), inputs(QWEN3_8B, { precision: 'FP8' as 'FP16' }))
      throw new Error('the call should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(InferenceInputError)
      expect((error as InferenceInputError).field).toBe('precision')
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

  it('names the two byte precision in the assumptions', () => {
    const result = estimateInference(qwen3(), inputs(QWEN3_8B))
    expect(
      result.assumptions.some((line) => line.includes('FP16') && line.includes('BF16')),
    ).toBe(true)
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
