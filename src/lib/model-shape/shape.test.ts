import { describe, expect, it } from 'vitest'

import type { RawConfig } from '../model-config'
import { loadConfigFixture } from '@/test/fixtures'

import { detectModelShape } from './shape'
import { detectMoeShape } from '@/lib/reap'
import { detectDsparkShape } from '@/lib/dspark'

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

describe('detectModelShape on a per layer expert flag list', () => {
  it('counts the expert blocks from the moe_layer_freq array', () => {
    // MiMo-V2.6-Flash-RL: 48 blocks, the first is dense, so 47 hold an expert bank.
    const flash = detectModelShape(loadConfigFixture('mimo-v26-flash-rl'))
    expect(flash?.moeLayers).toBe(47)
    expect(flash?.denseLayers).toBe(1)
    expect(flash?.routedExperts).toBe(256)
    expect(flash?.expertsPerToken).toBe(8)

    // MiMo-V2.6-Pro-RL: 70 blocks, the first is dense, so 69 hold an expert bank.
    const pro = detectModelShape(loadConfigFixture('mimo-v26-pro-rl'))
    expect(pro?.moeLayers).toBe(69)
    expect(pro?.denseLayers).toBe(1)
    expect(pro?.routedExperts).toBe(384)
  })

  it('reads a boolean array the same way as a zero and one array', () => {
    const base: RawConfig = {
      model_type: 'mimo_v2',
      hidden_size: 128,
      num_hidden_layers: 4,
      intermediate_size: 256,
      moe_intermediate_size: 64,
      n_routed_experts: 8,
      num_experts_per_tok: 2,
      vocab_size: 256,
      tie_word_embeddings: false,
    }
    const numeric = detectModelShape({ ...base, moe_layer_freq: [0, 1, 1, 1] })
    const boolean = detectModelShape({ ...base, moe_layer_freq: [false, true, true, true] })
    expect(numeric?.moeLayers).toBe(3)
    expect(boolean?.moeLayers).toBe(3)
    expect(boolean?.totalParams).toBe(numeric?.totalParams)
  })

  it('notes when the flag list does not cover every layer', () => {
    const shape = detectModelShape({
      model_type: 'mimo_v2',
      hidden_size: 128,
      num_hidden_layers: 4,
      intermediate_size: 256,
      moe_intermediate_size: 64,
      n_routed_experts: 8,
      num_experts_per_tok: 2,
      vocab_size: 256,
      tie_word_embeddings: false,
      moe_layer_freq: [0, 1],
    })
    expect(shape?.moeLayers).toBe(1)
    expect(shape?.notes.some((note) => note.includes('moe_layer_freq has 2 entries'))).toBe(true)
  })
})

describe('detectModelShape on a hybrid attention mixture of experts', () => {
  /**
   * MiMo-V2.6-Flash-RL, hand worked from the fixture.
   *
   * hidden 4096, 48 blocks, dense width 16384, vocabulary 152576, untied. The
   * hybrid layer pattern holds 9 global attention blocks and 39 sliding window
   * blocks. Both carry 64 query heads at a key and value width of 192 and a
   * value width of 128; the global blocks have 4 key and value heads and the
   * sliding ones have 8. The expert bank holds 256 experts at a width of 2048,
   * top-8, on 47 of the 48 blocks.
   *
   *   global block   4096*64*192 + 4096*4*192 + 4096*4*128 + 64*128*4096 =  89,128,960
   *   sliding block  4096*64*192 + 4096*8*192 + 4096*8*128 + 64*128*4096 =  94,371,840
   *   attention      9*89,128,960 + 39*94,371,840                        = 4,482,662,400
   *   dense feed fwd 1*3*4096*16384                                      =   201,326,592
   *   routed experts 47*256*3*4096*2048                                  = 302,795,194,368
   *   router         47*4096*256                                         =    49,283,072
   *   embedding and head  152576*4096*2                                  = 1,249,902,592
   *   total                                                              = 308,778,369,024
   *
   * The model card publishes 309B total and 15B activated parameters, so the
   * sum is checked against the release rather than against itself.
   */
  const FLASH_ATTENTION_PER_BLOCK = 89_128_960
  const FLASH_ATTENTION = 4_482_662_400
  const FLASH_TOTAL = 308_778_369_024
  const FLASH_ACTIVE = 14_146_338_816

  const PRO_ATTENTION = 18_717_081_600
  const PRO_TOTAL = 1_021_247_225_856
  const PRO_ACTIVE = 39_856_373_760

  it('sizes the global and sliding blocks with their own head geometry', () => {
    const flash = detectModelShape(loadConfigFixture('mimo-v26-flash-rl'))
    expect(flash?.attentionParamsPerLayer).toBe(FLASH_ATTENTION_PER_BLOCK)
    expect(flash?.attentionParams).toBe(FLASH_ATTENTION)
    expect(flash?.totalParams).toBe(FLASH_TOTAL)
    expect(flash?.activeParamsPerToken).toBe(FLASH_ACTIVE)
    expect(flash?.moeLayers).toBe(47)

    const pro = detectModelShape(loadConfigFixture('mimo-v26-pro-rl'))
    expect(pro?.attentionParams).toBe(PRO_ATTENTION)
    expect(pro?.totalParams).toBe(PRO_TOTAL)
    expect(pro?.activeParamsPerToken).toBe(PRO_ACTIVE)
  })

  it('matches the published parameter counts to rounding', () => {
    const flash = detectModelShape(loadConfigFixture('mimo-v26-flash-rl'))
    expect((flash?.totalParams ?? 0) / 1e9).toBeCloseTo(309, 0)

    const pro = detectModelShape(loadConfigFixture('mimo-v26-pro-rl'))
    expect((pro?.totalParams ?? 0) / 1e12).toBeCloseTo(1.02, 2)
  })

  it('reports an active path below the card figure, which counts more terms', () => {
    // The cards publish 15B and 42B activated parameters. This calculator counts
    // the attention path, the dense block, and the routed experts a token reads.
    // The card figures are larger because they also count the embedding table
    // and the multi token prediction blocks, which are not part of one forward
    // pass through the backbone. Both figures land in the same bracket.
    const flash = detectModelShape(loadConfigFixture('mimo-v26-flash-rl'))
    const flashActive = flash?.activeParamsPerToken ?? 0
    expect(flashActive / 1e9).toBeGreaterThan(14)
    expect(flashActive / 1e9).toBeLessThan(15)
    expect((flashActive + (flash?.embedParams ?? 0)) / 1e9).toBeCloseTo(15, 0)

    const pro = detectModelShape(loadConfigFixture('mimo-v26-pro-rl'))
    const proActive = pro?.activeParamsPerToken ?? 0
    expect(proActive / 1e9).toBeGreaterThan(39)
    expect(proActive / 1e9).toBeLessThan(42)
    expect((proActive + (pro?.embedParams ?? 0)) / 1e9).toBeCloseTo(42, 0)
  })

  it('agrees with the REAP and DSpark detectors on the same attention count', () => {
    const flash = loadConfigFixture('mimo-v26-flash-rl')
    expect(detectMoeShape(flash)?.attentionParams).toBe(FLASH_ATTENTION)
    expect(detectMoeShape(flash)?.totalParams).toBe(FLASH_TOTAL)
    expect(detectDsparkShape(flash)?.attentionParamsPerLayer).toBe(FLASH_ATTENTION_PER_BLOCK)
    expect(detectDsparkShape(flash)?.totalParams).toBe(FLASH_TOTAL)
  })

  it('keeps the value head width out of a model that does not set one', () => {
    // Qwen3-8B has no v_head_dim, so the value width stays at head_dim and the
    // block is the same 41,943,040 it always was.
    const shape = detectModelShape(loadConfigFixture('qwen3-8b'))
    expect(shape?.attentionParamsPerLayer).toBe(41_943_040)
    expect(shape?.attentionParams).toBe(36 * 41_943_040)
    expect(shape?.totalParams).toBe(QWEN3_8B_TOTAL_PARAMS)
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
