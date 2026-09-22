import { describe, expect, it } from 'vitest'

import { loadConfigFixture } from '@/test/fixtures'

import { computeKvCache, detectArchitecture, unwrapConfig } from './compute'
import { DTYPES, dtypeBytes, dtypeSupport, supportFor } from './dtypes'
import { KvCacheInputError, type ComputeResult, type RawConfig } from './types'

const fixture = loadConfigFixture

function total(name: string, options: Parameters<typeof computeKvCache>[1]): number {
  return computeKvCache(fixture(name), options).totalBytes
}

const GiB = 1024 ** 3

describe('hard anchors', () => {
  it('sizes Qwen3-8B at 32k context, one sequence, BF16', () => {
    // 2 x 8 kv heads x 128 head dim x 2 bytes = 4096 bytes per token per layer
    // 4096 x 36 layers x 32768 tokens = 4,831,838,208 bytes
    const result = computeKvCache(fixture('qwen3-8b'), {
      contextLength: 32768,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
    })

    expect(result.totalBytes).toBe(4_831_838_208)
    expect(result.totalBytes / GiB).toBeCloseTo(4.5, 6)
    expect(result.architecture.family).toBe('gqa')
    expect(result.architecture.modelType).toBe('qwen3')
    expect(result.layerSplit.total).toBe(36)
  })

  it('sizes DeepSeek-R1 at 128k context, one sequence, BF16', () => {
    // (512 kv_lora_rank + 64 qk_rope_head_dim) x 2 bytes = 1152 bytes per token per layer
    // 1152 x 61 layers x 131072 tokens = 9,210,691,584 bytes
    const result = computeKvCache(fixture('deepseek-r1'), {
      contextLength: 131072,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
    })

    expect(result.totalBytes).toBe(9_210_691_584)
    expect(result.totalBytes / GiB).toBeCloseTo(8.5781, 3)
    expect(result.architecture.family).toBe('mla')
    expect(result.layerSplit.total).toBe(61)
  })

  it('documents why the DeepSeek-R1 anchor is not 9,208,938,496', () => {
    // The brief listed 9,208,938,496 bytes for this case, but that value is not
    // reachable from its own stated derivation: 1152 x 61 x 131072 is exactly
    // 9,210,691,584. The formula was kept and the brief's figure corrected,
    // rather than bending the formula to hit a mistyped number.
    const perTokenPerLayer = (512 + 64) * 2
    const derived = perTokenPerLayer * 61 * 131072

    expect(derived).toBe(9_210_691_584)
    expect(derived).not.toBe(9_208_938_496)
    expect(total('deepseek-r1', { contextLength: 131072, kvCacheDtype: 'BF16' })).toBe(derived)
  })
})

describe('published cross checks', () => {
  it('matches the published 83.9 GiB for DeepSeek-V3.2-Exp at 1M context in BF16', () => {
    // vLLM reports 83.9 GiB for a 61 layer DeepSeek-V3.2 stack at 1M context
    // with a bf16 cache: MLA 1152 bytes plus a 256 byte indexer key per token.
    const result = computeKvCache(fixture('deepseek-v32-exp'), {
      contextLength: 1_048_576,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })

    expect(result.totalBytes).toBe(90_060_095_488)
    expect(result.totalBytes / GiB).toBeCloseTo(83.875, 3)
    // Within a tenth of a percent of the published figure.
    expect(Math.abs(result.totalBytes / GiB - 83.9) / 83.9).toBeLessThan(0.001)

    const indexer = result.components.find((component) => component.id === 'indexer')
    expect(indexer).toBeDefined()
    expect(indexer?.bytesPerToken).toBe(61 * 256)
  })

  it('lands within 1 percent of the published 9.62 GiB for DeepSeek-V4-Pro with an FP8 indexer', () => {
    // DeepSeek's published figure implies roughly one byte per indexer element,
    // and their indexer runs in FP8. With the indexer left at BF16 this
    // calculator reports about 10.65 GiB, which the assumptions explain.
    const withFp8Indexer = computeKvCache(fixture('deepseek-v4-pro'), {
      contextLength: 1_048_576,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
      indexerDtype: 'FP8_E4M3',
    })

    expect(withFp8Indexer.totalBytes / GiB).toBeCloseTo(9.6777, 3)
    expect(Math.abs(withFp8Indexer.totalBytes / GiB - 9.62) / 9.62).toBeLessThan(0.01)

    const withBf16Indexer = computeKvCache(fixture('deepseek-v4-pro'), {
      contextLength: 1_048_576,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })

    expect(withBf16Indexer.totalBytes / GiB).toBeCloseTo(10.6455, 3)
    expect(withBf16Indexer.assumptions.join(' ')).toMatch(/indexer/i)
  })

  it('sizes DeepSeek-V4-Pro from its per layer compress ratios', () => {
    const result = computeKvCache(fixture('deepseek-v4-pro'), {
      contextLength: 1_048_576,
      kvCacheDtype: 'BF16',
      indexerDtype: 'FP8_E4M3',
    })

    expect(result.architecture.family).toBe('dsv4')
    // 30 layers at ratio 4 and 31 at ratio 128, no uncompressed layers.
    expect(result.layerSplit.compressed).toEqual([
      { ratio: 4, layers: 30 },
      { ratio: 128, layers: 31 },
    ])
  })

  it('ignores the extra compress ratio entry that covers the MTP layer', () => {
    const config = fixture('deepseek-v4-flash')
    const ratios = config.compress_ratios as number[]

    // The array is one longer than the layer count in current V4 configs.
    expect(ratios.length).toBe(44)
    expect(config.num_hidden_layers).toBe(43)

    const result = computeKvCache(config, { contextLength: 1_048_576, kvCacheDtype: 'BF16' })
    expect(result.layerSplit.total).toBe(43)
    expect(result.layerSplit.compressed).toEqual([
      { ratio: 0, layers: 2 },
      { ratio: 4, layers: 21 },
      { ratio: 128, layers: 20 },
    ])
    expect(result.assumptions.join(' ')).toMatch(/multi token prediction layer/i)
  })
})

describe('sliding window layers', () => {
  it('caps sliding layers at the window instead of the context length', () => {
    const result = computeKvCache(fixture('gpt-oss-120b'), {
      contextLength: 131072,
      kvCacheDtype: 'BF16',
    })

    // 18 sliding layers hold only 128 entries each, 18 full layers hold 131072.
    expect(result.layerSplit.slidingAttention).toBe(18)
    expect(result.layerSplit.fullAttention).toBe(18)

    const perEntry = 2 * 8 * 64 * 2
    const expected = 18 * perEntry * 128 + 18 * perEntry * 131072
    expect(result.totalBytes).toBe(expected)
    expect(result.totalBytes).toBe(4_836_556_800)
  })

  it('treats every layer as sliding when the model applies one window everywhere', () => {
    const config: RawConfig = {
      model_type: 'mistral',
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      sliding_window: 4096,
    }

    const result = computeKvCache(config, { contextLength: 32768, kvCacheDtype: 'BF16' })
    const perEntry = 2 * 8 * 128 * 2
    expect(result.totalBytes).toBe(32 * perEntry * 4096)
    expect(result.layerSplit.slidingAttention).toBe(32)
  })
})

describe('a hybrid global and sliding window backbone', () => {
  /**
   * MiMo-V2.6-Flash-RL, hand worked from the fixture.
   *
   * 48 blocks: hybrid_layer_pattern marks 9 as global attention and 39 as
   * sliding window. The global blocks hold 4 key and value heads, the sliding
   * blocks hold 8, and both store a 192 wide key and a 128 wide value. The
   * window is 128 tokens.
   *
   *   global entry  4*(192+128)*2 = 2560 bytes
   *   global        9 * 2560 * 32768 =   754,974,720
   *   sliding entry 8*(192+128)*2 = 5120 bytes
   *   sliding       39 * 5120 * 128  =    25,559,040
   *   total                          =   780,533,760
   */
  it('sizes the global layers at the context and the sliding layers at the window', () => {
    const result = computeKvCache(fixture('mimo-v26-flash-rl'), {
      contextLength: 32768,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
    })

    expect(result.architecture.family).toBe('gqa')
    expect(result.architecture.modelType).toBe('mimo_v2')
    expect(result.bestEffort).toBe(false)
    expect(result.layerSplit.total).toBe(48)
    expect(result.layerSplit.fullAttention).toBe(9)
    expect(result.layerSplit.slidingAttention).toBe(39)

    const global = 9 * 4 * (192 + 128) * 2 * 32768
    const sliding = 39 * 8 * (192 + 128) * 2 * 128
    expect(result.totalBytes).toBe(global + sliding)
    expect(result.totalBytes).toBe(780_533_760)
    expect(result.assumptions.join(' ')).toMatch(/hybrid_layer_pattern/)
  })

  it('sizes MiMo-V2.6-Pro-RL the same way', () => {
    const result = computeKvCache(fixture('mimo-v26-pro-rl'), {
      contextLength: 32768,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
    })

    expect(result.layerSplit.total).toBe(70)
    expect(result.layerSplit.fullAttention).toBe(10)
    expect(result.layerSplit.slidingAttention).toBe(60)

    const global = 10 * 8 * (192 + 128) * 2 * 32768
    const sliding = 60 * 8 * (192 + 128) * 2 * 128
    expect(result.totalBytes).toBe(global + sliding)
    expect(result.totalBytes).toBe(1_717_043_200)
  })

  it('grows only the global layers when the context grows', () => {
    const short = computeKvCache(fixture('mimo-v26-flash-rl'), {
      contextLength: 8192,
      kvCacheDtype: 'BF16',
    })
    const long = computeKvCache(fixture('mimo-v26-flash-rl'), {
      contextLength: 32768,
      kvCacheDtype: 'BF16',
    })
    const globalEntry = 4 * (192 + 128) * 2
    expect(long.totalBytes - short.totalBytes).toBe(9 * globalEntry * (32768 - 8192))
  })

  it('records the two geometries as constants', () => {
    const result = computeKvCache(fixture('mimo-v26-flash-rl'), { contextLength: 32768 })
    const keys = result.constants.map((entry) => entry.key)
    expect(keys).toContain('hybrid_layer_pattern')
    expect(keys).toContain('v_head_dim')
    expect(keys).toContain('sliding_window')
    expect(result.constants.find((entry) => entry.key === 'hybrid_layer_pattern')?.value).toBe(
      '9 global / 39 sliding',
    )
  })
})

describe('architecture specific handling', () => {
  it('counts a shared key and value vector once when attention_k_eq_v is set', () => {
    const result = computeKvCache(fixture('gemma-4-31b'), {
      contextLength: 262144,
      kvCacheDtype: 'BF16',
    })

    const sliding = 50
    const full = 10
    const slidingEntry = 1 * 16 * 256 * 2
    const fullEntry = 1 * 4 * 512 * 2
    const expected = sliding * slidingEntry * 1024 + full * fullEntry * 262144

    expect(result.totalBytes).toBe(expected)
    expect(result.totalBytes).toBe(11_156_848_640)
    expect(result.assumptions.join(' ')).toMatch(/attention_k_eq_v/)
  })

  it('only counts the indexer cache on layers that store their own', () => {
    const result = computeKvCache(fixture('glm-5-3'), {
      contextLength: 131072,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })

    const mla = (512 + 64) * 2
    const indexerPerLayer = 128 * 2
    const fullIndexerLayers = 21
    const expected = (mla * 78 + indexerPerLayer * fullIndexerLayers) * 131072

    expect(result.totalBytes).toBe(expected)
    expect(result.totalBytes).toBe(12_482_248_704)

    const indexer = result.components.find((component) => component.id === 'indexer')
    expect(indexer?.bytesPerToken).toBe(indexerPerLayer * fullIndexerLayers)
    expect(indexer?.note).toMatch(/21 of 78/)
  })

  it('splits hybrid layers and keeps the linear state constant', () => {
    const result = computeKvCache(fixture('qwen3-next-80b'), {
      contextLength: 262144,
      kvCacheDtype: 'BF16',
    })

    expect(result.architecture.family).toBe('hybrid_linear')
    expect(result.layerSplit.fullAttention).toBe(12)
    expect(result.layerSplit.linearAttention).toBe(36)

    const perEntry = 2 * 2 * 256 * 2
    const recurrent = 32 * 128 * 128
    const conv = (16 * 128 + 2 * 32 * 128) * 4
    const expected = 12 * perEntry * 262144 + (recurrent + conv) * 36 * 2

    expect(result.totalBytes).toBe(expected)

    // The state must not change when the context grows.
    const shorter = computeKvCache(fixture('qwen3-next-80b'), {
      contextLength: 1024,
      kvCacheDtype: 'BF16',
    })
    const stateOf = (r: ComputeResult) => r.components.find((c) => c.id === 'state')?.bytesPerSequence
    expect(stateOf(shorter)).toBe(stateOf(result))
  })

  it('unwraps a nested text_config', () => {
    const raw = fixture('deepseek-v41-flash')
    const { inner, outer } = unwrapConfig(raw)

    expect(inner).not.toBe(raw)
    expect(outer).toBe(raw)
    expect(inner.model_type).toBe('deepseek_v41_text')

    const result = computeKvCache(raw, { contextLength: 65536, kvCacheDtype: 'BF16' })
    expect(result.architecture.family).toBe('dsv41')
    expect(result.layerSplit.total).toBe(40)
  })
})

describe('DeepSeek V4.1', () => {
  it('detects V4.1 as its own family, separate from V4', () => {
    const v41 = computeKvCache(fixture('deepseek-v41-flash'), {
      contextLength: 1_048_576,
      kvCacheDtype: 'FP4',
      indexerDtype: 'FP4',
    })
    expect(v41.architecture.family).toBe('dsv41')
    expect(v41.bestEffort).toBe(false)
    expect(v41.architecture.modelType).toBe('deepseek_v41')

    for (const name of ['deepseek-v4-pro', 'deepseek-v4-flash']) {
      const v4 = computeKvCache(fixture(name), { contextLength: 4096, kvCacheDtype: 'BF16' })
      expect(v4.architecture.family, name).toBe('dsv4')
    }
  })

  it('matches the published 890 bytes per token at 1M context with an FP4 cache', () => {
    // Only kv_source_layer_ids (2, 8, 14, 20) own the global cache. Three of
    // them sit at ratio 2 and one at ratio 1.
    //   main KV    512 head_dim, rope dims included
    //              FP4: 256 payload + 32 scales (one E4M3 per 16) = 288 bytes
    //              288 / 2 * 3 + 288 = 720 bytes per token
    //   indexer K  128 index_head_dim
    //              FP4: 64 payload + 4 scales (one UE8M0 per 32) = 68 bytes
    //              68 / 2 * 3 + 68 = 170 bytes per token
    //   720 + 170 = 890 bytes per token
    const result = computeKvCache(fixture('deepseek-v41-flash'), {
      contextLength: 1_048_576,
      sequenceCount: 1,
      kvCacheDtype: 'FP4',
      indexerDtype: 'FP4',
    })

    expect(result.bytesPerToken).toBe(890)
    expect(result.totalBytes).toBe(933_232_640)
    expect(result.totalBytes / GiB).toBeLessThan(1)
    expect(
      result.components.find((component) => component.id === 'attention')?.bytesPerToken,
    ).toBe(720)
    expect(
      result.components.find((component) => component.id === 'indexer')?.bytesPerToken,
    ).toBe(170)
  })

  it('sizes only the source layers, not all 40', () => {
    const result = computeKvCache(fixture('deepseek-v41-flash'), {
      contextLength: 1_048_576,
      kvCacheDtype: 'FP4',
      indexerDtype: 'FP4',
    })

    expect(result.layerSplit.total).toBe(40)
    // One entry per source layer, at its own compression ratio.
    expect(result.layerSplit.compressed).toEqual([
      { ratio: 1, layers: 1 },
      { ratio: 2, layers: 3 },
    ])
    expect(result.assumptions.join(' ')).toMatch(/kv_source_layer_ids/)
    expect(result.assumptions.join(' ')).toMatch(/compress_ratios/)
  })

  it('carries the FP4 block scales instead of a bare half byte per element', () => {
    // A naive FP4 figure would be (512 + 128) * 0.5 * (3/2 + 1) = 640 bytes
    // per token. The scales are what turn that into the published 890.
    const result = computeKvCache(fixture('deepseek-v41-flash'), {
      contextLength: 1_048_576,
      kvCacheDtype: 'FP4',
      indexerDtype: 'FP4',
    })

    expect(result.bytesPerToken).toBeGreaterThan(640)
    expect(result.assumptions.join(' ')).toMatch(/E4M3 scale per 16/)
    expect(result.assumptions.join(' ')).toMatch(/UE8M0 scale per 32/)
  })

  it('marks FP4 as supported for both V4.1 dropdowns', () => {
    const result = computeKvCache(fixture('deepseek-v41-flash'), {
      contextLength: 1_048_576,
      kvCacheDtype: 'FP4',
      indexerDtype: 'FP4',
    })

    expect(result.dtypeSupport.find((entry) => entry.dtype === 'FP4')?.level).toBe('supported')
    expect(result.indexerDtypeSupport?.find((entry) => entry.dtype === 'FP4')?.level).toBe(
      'supported',
    )
    expect(supportFor('dsv41', 'kv', 'INT8').level).toBe('unsupported')
  })
})

describe('best effort handling', () => {
  it('flags an unrecognised model_type but still returns a number when the shape is inferable', () => {
    const config: RawConfig = {
      model_type: 'some_brand_new_arch',
      num_hidden_layers: 24,
      num_attention_heads: 16,
      num_key_value_heads: 4,
      head_dim: 64,
    }

    const result = computeKvCache(config, { contextLength: 8192, kvCacheDtype: 'BF16' })

    expect(result.bestEffort).toBe(true)
    expect(result.totalBytes).toBe(2 * 4 * 64 * 2 * 24 * 8192)
  })

  it('falls back to the MLA shape when kv_lora_rank is present', () => {
    const config: RawConfig = {
      model_type: 'unknown_mla_thing',
      num_hidden_layers: 10,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
    }

    const result = computeKvCache(config, { contextLength: 4096, kvCacheDtype: 'BF16' })
    expect(result.architecture.family).toBe('mla')
    expect(result.bestEffort).toBe(true)
    expect(result.totalBytes).toBe(1152 * 10 * 4096)
  })

  it('reports clearly when nothing in the config is recognisable', () => {
    const config: RawConfig = { model_type: 'mystery', vocab_size: 32000 }

    expect(() => computeKvCache(config, { contextLength: 4096 })).toThrow(KvCacheInputError)
    expect(() => computeKvCache(config, { contextLength: 4096 })).toThrow(/cannot be derived/i)
  })

  it('detects architectures from architectures[0] when model_type is generic', () => {
    const detection = detectArchitecture(
      { num_hidden_layers: 61, kv_lora_rank: 512, qk_rope_head_dim: 64 },
      { architectures: ['DeepseekV3ForCausalLM'] },
    )
    expect(detection.family).toBe('mla')
  })
})

describe('input guards', () => {
  it('rejects a zero, negative, or fractional context length', () => {
    for (const contextLength of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => computeKvCache(fixture('qwen3-8b'), { contextLength })).toThrow(KvCacheInputError)
    }
  })

  it('rejects a zero or fractional sequence count', () => {
    for (const sequenceCount of [0, -3, 2.5, Number.NaN]) {
      expect(() =>
        computeKvCache(fixture('qwen3-8b'), { contextLength: 4096, sequenceCount }),
      ).toThrow(KvCacheInputError)
    }
  })

  it('never returns NaN, Infinity, or a negative number across every fixture', () => {
    const names = [
      'qwen3-8b',
      'deepseek-r1',
      'deepseek-v32-exp',
      'deepseek-v4-pro',
      'deepseek-v4-flash',
      'deepseek-v41-flash',
      'qwen3-next-80b',
      'qwen3-5-27b',
      'glm-4-7-flash',
      'glm-5-3',
      'gpt-oss-120b',
      'gemma-4-31b',
      'nemotron-h-nano',
      'kimi-k2',
    ]

    for (const name of names) {
      for (const contextLength of [1, 4096, 131072]) {
        const result = computeKvCache(fixture(name), {
          contextLength,
          sequenceCount: 2,
          kvCacheDtype: 'BF16',
          indexerDtype: 'BF16',
        })

        expect(Number.isFinite(result.totalBytes), `${name} total`).toBe(true)
        expect(result.totalBytes).toBeGreaterThanOrEqual(0)
        expect(Number.isFinite(result.bytesPerToken), `${name} per token`).toBe(true)

        for (const component of result.components) {
          expect(Number.isFinite(component.totalBytes), `${name} ${component.id}`).toBe(true)
          expect(component.totalBytes).toBeGreaterThanOrEqual(0)
        }
      }
    }
  })

  it('rejects an unknown dtype with a clear message', () => {
    // Radix Select can report an empty value; the engine must not surface a
    // bare "Unknown dtype" error for it.
    expect(() =>
      computeKvCache(fixture('qwen3-8b'), {
        contextLength: 4096,
        kvCacheDtype: '' as never,
      }),
    ).toThrow(/Unknown kv_cache_dtype/)

    expect(() =>
      computeKvCache(fixture('deepseek-v32-exp'), {
        contextLength: 4096,
        indexerDtype: 'NOT_A_DTYPE' as never,
      }),
    ).toThrow(/Unknown indexer_dtype/)
  })

  it('scales linearly with sequence count', () => {
    const one = computeKvCache(fixture('qwen3-8b'), { contextLength: 32768, sequenceCount: 1 })
    const four = computeKvCache(fixture('qwen3-8b'), { contextLength: 32768, sequenceCount: 4 })
    expect(four.totalBytes).toBe(one.totalBytes * 4)
  })

  it('scales linearly with context length', () => {
    const small = computeKvCache(fixture('qwen3-8b'), { contextLength: 8192 })
    const large = computeKvCache(fixture('qwen3-8b'), { contextLength: 32768 })
    expect(large.totalBytes).toBe(small.totalBytes * 4)
  })

  it('halves the cache when the dtype halves in width', () => {
    const bf16 = computeKvCache(fixture('qwen3-8b'), { contextLength: 32768, kvCacheDtype: 'BF16' })
    const fp8 = computeKvCache(fixture('qwen3-8b'), {
      contextLength: 32768,
      kvCacheDtype: 'FP8_E4M3',
    })
    expect(fp8.totalBytes).toBe(bf16.totalBytes / 2)
  })

  it('warns when the context exceeds the model limit', () => {
    const result = computeKvCache(fixture('qwen3-8b'), { contextLength: 1_000_000 })
    expect(result.maxPositionEmbeddings).toBe(40960)
    expect(result.assumptions.join(' ')).toMatch(/max_position_embeddings/)
  })
})

describe('breakdown output', () => {
  it('reports the formula, the constants used, and component subtotals', () => {
    const result = computeKvCache(fixture('deepseek-v32-exp'), {
      contextLength: 131072,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })

    expect(result.components.map((component) => component.id)).toEqual(['attention', 'indexer'])

    const attention = result.components[0]
    expect(attention.formula).toContain('latent attention layers')
    expect(attention.bytesPerToken).toBe(1152 * 61)

    const indexer = result.components[1]
    expect(indexer.formula).toContain('index_head_dim')

    // Component subtotals must add up to the total.
    const sum = result.components.reduce((acc, component) => acc + component.totalBytes, 0)
    expect(sum).toBe(result.totalBytes)

    // Per token contributions must also add up.
    const perToken = result.components.reduce((acc, component) => acc + component.bytesPerToken, 0)
    expect(perToken).toBeCloseTo(result.bytesPerToken, 6)

    const keys = result.constants.map((constant) => constant.key)
    expect(keys).toContain('num_hidden_layers')
    expect(keys).toContain('kv_lora_rank')
    expect(keys).toContain('index_head_dim')

    expect(result.steps.length).toBeGreaterThan(3)
    expect(result.assumptions.join(' ')).toMatch(/scale factor/i)
  })

  it('reports the layer split for every family', () => {
    for (const name of ['qwen3-8b', 'deepseek-r1', 'qwen3-next-80b', 'deepseek-v4-pro']) {
      const result = computeKvCache(fixture(name), { contextLength: 4096 })
      expect(result.layerSplit.total).toBeGreaterThan(0)
    }
  })
})

describe('dtype catalog', () => {
  it('covers all eight dtypes with the right widths', () => {
    expect(DTYPES).toHaveLength(8)
    expect(dtypeBytes('BF16')).toBe(2)
    expect(dtypeBytes('FP16')).toBe(2)
    expect(dtypeBytes('FP32')).toBe(4)
    expect(dtypeBytes('FP8_E4M3')).toBe(1)
    expect(dtypeBytes('FP8_E5M2')).toBe(1)
    expect(dtypeBytes('INT8')).toBe(1)
    expect(dtypeBytes('INT4')).toBe(0.5)
    expect(dtypeBytes('FP4')).toBe(0.5)
  })

  it('tags every dtype for every family with a reason', () => {
    for (const family of ['gqa', 'mla', 'hybrid_linear', 'dsv4', 'unknown'] as const) {
      for (const role of ['kv', 'indexer'] as const) {
        const support = dtypeSupport(family, role)
        expect(support).toHaveLength(8)
        for (const entry of support) {
          expect(['supported', 'untested', 'unsupported']).toContain(entry.level)
          expect(entry.reason.length).toBeGreaterThan(10)
        }
      }
    }
  })

  it('grounds the notable tags in real behaviour', () => {
    // FP8 KV cache is a documented vLLM and SGLang option.
    expect(supportFor('gqa', 'kv', 'FP8_E4M3').level).toBe('supported')
    // The DeepSeek-V3.2 indexer stores block scaled FP8 keys.
    expect(supportFor('mla', 'indexer', 'FP8_E4M3').level).toBe('supported')
    // V4 deployments run the indexer cache in FP4.
    expect(supportFor('dsv4', 'indexer', 'FP4').level).toBe('supported')
    // No engine exposes an FP32 cache.
    expect(supportFor('gqa', 'kv', 'FP32').level).toBe('unsupported')
    // BF16 is the default everywhere.
    expect(supportFor('dsv4', 'kv', 'BF16').level).toBe('supported')
  })

  it('defaults to BF16', () => {
    const result = computeKvCache(fixture('qwen3-8b'), { contextLength: 32768 })
    const explicit = computeKvCache(fixture('qwen3-8b'), {
      contextLength: 32768,
      kvCacheDtype: 'BF16',
    })
    expect(result.totalBytes).toBe(explicit.totalBytes)
  })

  it('exposes the attention tags and the indexer tags separately', () => {
    const withIndexer = computeKvCache(fixture('deepseek-v32-exp'), { contextLength: 4096 })
    expect(withIndexer.dtypeSupport).toHaveLength(8)
    expect(withIndexer.indexerDtypeSupport).toHaveLength(8)
    // The indexer tags describe the indexer, not the attention cache.
    expect(
      withIndexer.indexerDtypeSupport?.find((entry) => entry.dtype === 'FP8_E4M3')?.level,
    ).toBe('supported')

    const withoutIndexer = computeKvCache(fixture('qwen3-8b'), { contextLength: 4096 })
    expect(withoutIndexer.dtypeSupport).toHaveLength(8)
    // No indexer means no indexer dtype field at all.
    expect(withoutIndexer.indexerDtypeSupport).toBeNull()
  })
})
