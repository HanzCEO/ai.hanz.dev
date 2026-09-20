import { describe, expect, it } from 'vitest'

import type { RawConfig } from '../model-config'
import { loadConfigFixture } from '@/test/fixtures'

import { detectModelShape } from './shape'

/**
 * Qwen3-8B, hand worked from the fixture.
 *
 * hidden 4096, 36 blocks, dense width 12288, vocabulary 151936, untied, 32
 * attention heads and 8 key and value heads at a head width of 128.
 *
 *   attention block  4096*32*128 + 2*4096*8*128 + 32*128*4096 = 41,943,040
 *   attention        36 * 41,943,040                          = 1,509,949,440
 *   dense feed fwd   36 * 3 * 4096 * 12288                    = 5,435,817,984
 *   embedding and head  151936 * 4096 * 2                     = 1,244,659,712
 *   total                                                     = 8,190,427,136
 *
 * The last line is the published 8.19 billion parameter count, so the sum is
 * checked against the model rather than against itself.
 */
const QWEN3_8B_TOTAL_PARAMS = 8_190_427_136

/**
 * A sparse mixture of experts, hand worked from the fields below.
 *
 * 23 expert blocks and 1 dense block. 128 routed experts at a width of 768,
 * top-8, 2 always-on shared experts at a width of 768.
 */
const MOE: RawConfig = {
  model_type: 'qwen3_moe',
  hidden_size: 2048,
  num_hidden_layers: 24,
  num_attention_heads: 32,
  num_key_value_heads: 4,
  head_dim: 128,
  intermediate_size: 6144,
  moe_intermediate_size: 768,
  num_experts: 128,
  num_experts_per_tok: 8,
  n_shared_experts: 2,
  shared_expert_intermediate_size: 768,
  first_k_dense_replace: 1,
  vocab_size: 151936,
  tie_word_embeddings: false,
}

const MOE_LAYERS = 23
const MOE_ROUTED_EXPERT_PARAMS = 23 * 128 * 3 * 2048 * 768
const MOE_SHARED_EXPERT_PARAMS = 23 * 2 * 3 * 2048 * 768
const MOE_TOTAL_PARAMS =
  24 * (2048 * 32 * 128 + 2 * 2048 * 4 * 128 + 32 * 128 * 2048) +
  1 * 3 * 2048 * 6144 +
  MOE_ROUTED_EXPERT_PARAMS +
  MOE_SHARED_EXPERT_PARAMS +
  23 * 2048 * 128 +
  151936 * 2048 * 2

describe('detectModelShape on a dense model', () => {
  it('counts every weight the checkpoint holds', () => {
    const shape = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(shape).not.toBeNull()
    expect(shape?.totalParams).toBe(QWEN3_8B_TOTAL_PARAMS)
  })

  it('reports no expert bank', () => {
    const shape = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(shape?.moeLayers).toBe(0)
    expect(shape?.denseLayers).toBe(36)
    expect(shape?.routedExperts).toBe(0)
    expect(shape?.routedExpertParams).toBe(0)
    expect(shape?.sharedExpertParams).toBe(0)
    expect(shape?.routerParams).toBe(0)
  })

  it('counts the language model head separately when the embedding is untied', () => {
    const shape = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(shape?.tiedEmbeddings).toBe(false)
    expect(shape?.embedParams).toBe(151936 * 4096 * 2)
  })

  it('counts one embedding table when the config ties it', () => {
    const tied = detectModelShape({ ...loadConfigFixture('qwen3-8b'), tie_word_embeddings: true })
    const untied = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(tied?.embedParams).toBe(151936 * 4096)
    expect((untied?.totalParams ?? 0) - (tied?.totalParams ?? 0)).toBe(151936 * 4096)
  })

  it('reads the attention heads and the head width', () => {
    const shape = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(shape?.numAttentionHeads).toBe(32)
    expect(shape?.numKeyValueHeads).toBe(8)
    expect(shape?.headDim).toBe(128)
    expect(shape?.attentionParamsPerLayer).toBe(41_943_040)
  })
})

describe('detectModelShape on a mixture of experts', () => {
  it('counts the expert blocks from the dense prefix', () => {
    expect(detectModelShape(MOE)?.moeLayers).toBe(MOE_LAYERS)
    expect(detectModelShape(MOE)?.denseLayers).toBe(1)
  })

  it('counts the routed experts, the shared experts and the router', () => {
    const shape = detectModelShape(MOE)
    expect(shape?.routedExpertParams).toBe(MOE_ROUTED_EXPERT_PARAMS)
    expect(shape?.sharedExpertParams).toBe(MOE_SHARED_EXPERT_PARAMS)
    expect(shape?.routerParams).toBe(23 * 2048 * 128)
  })

  it('sums to the hand-worked total', () => {
    const shape = detectModelShape(MOE)
    expect(shape?.totalParams).toBe(MOE_TOTAL_PARAMS)

    const parts = [
      shape?.attentionParams,
      shape?.denseFfnParams,
      shape?.routedExpertParams,
      shape?.sharedExpertParams,
      shape?.routerParams,
      shape?.embedParams,
    ]
    expect(parts.reduce<number>((sum, part) => sum + (part ?? 0), 0)).toBe(MOE_TOTAL_PARAMS)
  })

  it('counts fewer active parameters than resident parameters', () => {
    const shape = detectModelShape(MOE)
    // The sparse path is the point of the architecture, so the active count
    // must not equal the resident count.
    expect(shape?.activeParamsPerToken).toBeLessThan(shape?.totalParams ?? 0)
  })

  it('reads the shared expert width from the config', () => {
    const shape = detectModelShape(MOE)
    expect(shape?.sharedExperts).toBe(2)
    expect(shape?.sharedExpertIntermediate).toBe(768)
  })
})

describe('detectModelShape on a config that is not a language model', () => {
  it('returns null when the hidden size is absent', () => {
    expect(detectModelShape({})).toBeNull()
    expect(detectModelShape({ num_hidden_layers: 12 })).toBeNull()
  })

  it('returns null when the depth is absent', () => {
    expect(detectModelShape({ hidden_size: 4096 })).toBeNull()
  })
})

describe('detectModelShape best effort flags', () => {
  it('flags a missing vocabulary and a missing tie setting', () => {
    const shape = detectModelShape({ hidden_size: 4096, num_hidden_layers: 12 })
    expect(shape?.bestEffort).toBe(true)
    expect(shape?.notes.some((note) => note.includes('vocab_size'))).toBe(true)
    expect(shape?.notes.some((note) => note.includes('tie_word_embeddings'))).toBe(true)
  })

  it('flags an inferred dense width', () => {
    const shape = detectModelShape({
      hidden_size: 4096,
      num_hidden_layers: 12,
      vocab_size: 32000,
      tie_word_embeddings: true,
    })
    expect(shape?.intermediateSize).toBe(4 * 4096)
    expect(shape?.notes.some((note) => note.includes('intermediate_size'))).toBe(true)
  })

  it('carries no note for a complete config', () => {
    const shape = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(shape?.bestEffort).toBe(false)
    expect(shape?.notes).toEqual([])
  })
})

describe('detectModelShape on a nested multimodal config', () => {
  it('reads the language model fields from the nested config', () => {
    const shape = detectModelShape({
      model_type: 'multimodal',
      text_config: {
        model_type: 'qwen3',
        hidden_size: 4096,
        num_hidden_layers: 36,
        intermediate_size: 12288,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 151936,
        tie_word_embeddings: false,
      },
    })
    expect(shape?.modelType).toBe('qwen3')
    expect(shape?.totalParams).toBe(QWEN3_8B_TOTAL_PARAMS)
  })
})
