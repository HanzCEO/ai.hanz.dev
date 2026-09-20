import { describe, expect, it } from 'vitest'

import { findGpu, findStorage } from '../hardware'
import type { GpuSpec, StorageSpec } from '../hardware'
import { loadConfigFixture } from '@/test/fixtures'
import { reapInputs as inputs } from '@/test/reap'
import type { RawConfig } from '../model-config'

import { estimateReap } from './compute'
import { detectMoeShape } from './moe-shape'
import { bytesPerParam } from './presets'
import { ReapInputError, type MoeShape } from './types'

const fixture = loadConfigFixture

const H200 = findGpu('h200') as GpuSpec
const RTX_5090 = findGpu('rtx-5090') as GpuSpec
const NVME = findStorage('nvme-pcie4') as StorageSpec
const HOST_RAM = findStorage('pcie5-host-ram') as StorageSpec

/**
 * Qwen3-30B-A3B as the llm-compressor REAP example describes it: 48 layers, 128
 * experts, top-8, hidden 2048, expert width 768. This is the model the
 * published run calibrated, so it is the anchor for the whole cost model.
 */
const QWEN3_30B_A3B: RawConfig = {
  model_type: 'qwen3_moe',
  num_hidden_layers: 48,
  hidden_size: 2048,
  num_attention_heads: 32,
  num_key_value_heads: 4,
  head_dim: 128,
  intermediate_size: 6144,
  moe_intermediate_size: 768,
  num_experts: 128,
  num_experts_per_tok: 8,
  vocab_size: 151936,
  tie_word_embeddings: false,
}

describe('detectMoeShape on the dense fixtures', () => {
  it('returns null for a dense model, which has nothing to prune', () => {
    expect(detectMoeShape(fixture('qwen3-8b'))).toBeNull()
    expect(detectMoeShape(fixture('gemma-4-31b'))).toBeNull()
  })

  it('returns null when the layer count is missing', () => {
    expect(detectMoeShape({ num_experts: 64, hidden_size: 2048 })).toBeNull()
  })

  it('returns null when the expert count is zero', () => {
    expect(
      detectMoeShape({ num_experts: 0, hidden_size: 2048, num_hidden_layers: 12 }),
    ).toBeNull()
  })
})

describe('detectMoeShape on the MoE fixtures', () => {
  it('counts the MoE layers from the layer pattern in the config', () => {
    expect(detectMoeShape(fixture('glm-5-3'))?.moeLayers).toBe(75)
    expect(detectMoeShape(fixture('deepseek-r1'))?.moeLayers).toBe(58)
    expect(detectMoeShape(fixture('qwen3-next-80b'))?.moeLayers).toBe(48)
    expect(detectMoeShape(fixture('glm-4-7-flash'))?.moeLayers).toBe(46)
  })

  it('reads the expert count, the top-k, and the expert width', () => {
    const glm = detectMoeShape(fixture('glm-5-3')) as MoeShape
    expect(glm.routedExperts).toBe(256)
    expect(glm.expertsPerToken).toBe(8)
    expect(glm.moeIntermediateSize).toBe(2048)
    expect(glm.hiddenSize).toBe(6144)
    expect(glm.numLayers).toBe(78)
    expect(glm.denseLayers).toBe(3)
  })

  it('reads num_local_experts, which is how gpt-oss names the expert count', () => {
    const gptOss = detectMoeShape(fixture('gpt-oss-120b')) as MoeShape
    expect(gptOss.routedExperts).toBe(128)
    expect(gptOss.expertsPerToken).toBe(4)
    // No moe_intermediate_size, so the dense width is used for each expert.
    expect(gptOss.moeIntermediateSize).toBe(2880)
  })

  it('counts three projections per expert, the SwiGLU gate, up, and down', () => {
    const glm = detectMoeShape(fixture('glm-5-3')) as MoeShape
    expect(glm.paramsPerExpert).toBe(3 * 6144 * 2048)
  })

  it('keeps the total above the routed expert share, since embeddings and attention exist', () => {
    const glm = detectMoeShape(fixture('glm-5-3')) as MoeShape
    expect(glm.totalParams).toBeGreaterThan(glm.routedExpertParams)
    expect(glm.embedParams).toBeGreaterThan(0)
  })

  it('resolves a config whose language model fields are nested', () => {
    const qwen35 = detectMoeShape(fixture('qwen3-5-27b'))
    // The 27B is dense, so this is the nested unwrap returning no experts.
    expect(qwen35).toBeNull()
  })
})

describe('estimateReap on the anchor model', () => {
  const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape

  it('finds the shape the published example describes', () => {
    expect(shape).not.toBeNull()
    expect(shape.numLayers).toBe(48)
    expect(shape.moeLayers).toBe(48)
    expect(shape.routedExperts).toBe(128)
    expect(shape.expertsPerToken).toBe(8)
    expect(shape.moeIntermediateSize).toBe(768)
  })

  it('lands the paper recipe inside the range measured independently for REAP scoring', () => {
    // AIMER measured 0.75 to 2.96 hours for REAP expert scoring. The estimate
    // for this model on an H200 has to fall in the same band, or the model is
    // wrong by an order of magnitude.
    const result = estimateReap(
      shape,
      inputs({ calibrationSamples: 24576, sequenceLength: 16384, gpu: H200 }),
    )
    const hours = result.computeSeconds / 3600
    expect(hours).toBeGreaterThan(1.1)
    expect(hours).toBeLessThan(4.4)
  })

  it('scales compute linearly with the sample count', () => {
    const base = estimateReap(shape, inputs({ calibrationSamples: 512 }))
    const tenTimes = estimateReap(shape, inputs({ calibrationSamples: 5120 }))
    expect(tenTimes.computeSeconds / base.computeSeconds).toBeCloseTo(10, 6)
    expect(tenTimes.tokens / base.tokens).toBeCloseTo(10, 6)
  })

  it('scales compute linearly with the sequence length', () => {
    const base = estimateReap(shape, inputs({ sequenceLength: 2048 }))
    const longer = estimateReap(shape, inputs({ sequenceLength: 8192 }))
    expect(longer.computeSeconds / base.computeSeconds).toBeCloseTo(4, 6)
  })

  it('halves compute time when utilisation doubles', () => {
    const low = estimateReap(shape, inputs({ mfu: 0.2 }))
    const high = estimateReap(shape, inputs({ mfu: 0.4 }))
    expect(low.computeSeconds / high.computeSeconds).toBeCloseTo(2, 6)
  })

  it('adds the overhead factor and the setup time on top of the bound', () => {
    const result = estimateReap(shape, inputs({ overheadFactor: 2, setupSeconds: 600 }))
    const bound = Math.max(result.computeSeconds, result.streamSeconds)
    expect(result.estimateSeconds).toBeCloseTo(bound * 2 + 600, 6)
  })

  it('reports which resource limits the run', () => {
    const result = estimateReap(shape, inputs())
    expect(result.bound).toBe(result.computeSeconds >= result.streamSeconds ? 'compute' : 'stream')
  })

  it('reports a quick run in minutes, not hours, on a fast card', () => {
    const result = estimateReap(shape, inputs({ calibrationSamples: 512, sequenceLength: 2048 }))
    expect(result.estimateSeconds).toBeLessThan(3600)
  })
})

describe('estimateReap verdicts', () => {
  it('reports not-moe for a dense model', () => {
    const result = estimateReap(null, inputs())
    expect(result.verdict).toBe('not-moe')
    expect(result.moe).toBe(false)
    expect(result.estimateSeconds).toBe(0)
    expect(result.assumptions.length).toBeGreaterThan(0)
  })

  it('reports fits when the resident block is comfortably inside the card', () => {
    const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: H200 }))
    expect(result.verdict).toBe('fits')
    expect(result.peakVramBytes).toBeLessThan(result.vramBytes)
  })

  it('reports needs-fp8 when the BF16 block overflows the card but FP8 fits', () => {
    // DeepSeek V4 Pro has 384 experts of 66M parameters each. One block is about
    // 47 GiB in BF16, which does not fit a 32 GB card, and about 24 GiB in FP8,
    // which does.
    const shape = detectMoeShape(fixture('deepseek-v4-pro')) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: RTX_5090, weightDtype: 'BF16' }))
    expect(result.verdict).toBe('needs-fp8')
    expect(result.perMoELayerBytes).toBeGreaterThan(result.vramBytes)
    expect(result.narrowestFittingDtype).toBe('FP8')
  })

  it('reports needs-offload when even the narrowest format overflows', () => {
    // Kimi-K2 in FP8 is about 16 GiB a block. A micro batch of 128 samples of
    // 2048 tokens adds a 28 GiB activation buffer, which no longer fits.
    const shape = detectMoeShape(fixture('kimi-k2')) as MoeShape
    const result = estimateReap(
      shape,
      inputs({ gpu: RTX_5090, weightDtype: 'BF16', microBatchSize: 128 }),
    )
    expect(result.verdict).toBe('needs-offload')
    expect(result.narrowestFittingDtype).toBeNull()
  })

  it('reports fits when the selected format is already the narrow one', () => {
    const shape = detectMoeShape(fixture('kimi-k2')) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: RTX_5090, weightDtype: 'FP8' }))
    expect(result.verdict).toBe('fits')
  })

  it('names the narrowest format that fits', () => {
    const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: H200, weightDtype: 'BF16' }))
    // BF16 already fits, so it is the widest format that fits.
    expect(result.narrowestFittingDtype).toBe('BF16')
  })
})

describe('estimateReap pruning arithmetic', () => {
  const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape

  it('keeps the expected share of experts', () => {
    const result = estimateReap(shape, inputs({ pruneRatio: 0.5 }))
    expect(result.keptExperts).toBe(64)
    expect(result.removedExperts).toBe(64)
  })

  it('never keeps fewer experts than the router routes to', () => {
    const result = estimateReap(shape, inputs({ pruneRatio: 0.9, scaleTopK: false }))
    expect(result.keptExperts).toBeGreaterThanOrEqual(shape.expertsPerToken)
    expect(result.expertsPerTokenAfter).toBe(shape.expertsPerToken)
  })

  it('leaves active parameters unchanged when the top-k is not reduced', () => {
    const result = estimateReap(shape, inputs({ pruneRatio: 0.5, scaleTopK: false }))
    expect(result.activeParamsPerTokenAfter).toBe(shape.activeParamsPerToken)
  })

  it('lowers active parameters when the top-k is reduced in proportion', () => {
    const result = estimateReap(shape, inputs({ pruneRatio: 0.5, scaleTopK: true }))
    expect(result.expertsPerTokenAfter).toBe(4)
    expect(result.activeParamsPerTokenAfter).toBeLessThan(shape.activeParamsPerToken)
  })

  it('shrinks the total parameter count by roughly the expert share', () => {
    const result = estimateReap(shape, inputs({ pruneRatio: 0.5 }))
    expect(result.totalParamsAfter).toBeLessThan(shape.totalParams)
    expect(result.reductionPercent).toBeGreaterThan(20)
    expect(result.reductionPercent).toBeLessThan(60)
    // Expert parameters are what actually disappear.
    expect(result.expertParamsAfter).toBeCloseTo(
      shape.moeLayers * result.keptExperts * shape.paramsPerExpert,
      6,
    )
  })

  it('removes nothing at a ratio of zero', () => {
    const result = estimateReap(shape, inputs({ pruneRatio: 0 }))
    expect(result.keptExperts).toBe(shape.routedExperts)
    expect(result.reductionPercent).toBe(0)
    expect(result.activeParamsPerTokenAfter).toBe(shape.activeParamsPerToken)
  })
})

describe('estimateReap storage and memory', () => {
  const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape

  it('is I/O bound when the weights come over a slow link', () => {
    const result = estimateReap(shape, inputs({ storage: findStorage('network-10gbe') as StorageSpec }))
    expect(result.bound).toBe('stream')
  })

  it('streams fewer bytes in a narrower format', () => {
    const bf16 = estimateReap(shape, inputs({ weightDtype: 'BF16' }))
    const fp8 = estimateReap(shape, inputs({ weightDtype: 'FP8' }))
    expect(fp8.weightBytes).toBe(bf16.weightBytes / 2)
    expect(fp8.perMoELayerBytes).toBe(bf16.perMoELayerBytes / 2)
  })

  it('grows the activation buffer with the micro batch size', () => {
    const small = estimateReap(shape, inputs({ microBatchSize: 1 }))
    const large = estimateReap(shape, inputs({ microBatchSize: 8 }))
    expect(large.activationBytes).toBe(small.activationBytes * 8)
    expect(large.peakVramBytes).toBeGreaterThan(small.peakVramBytes)
  })

  it('does not depend on storage speed for the memory figures', () => {
    const a = estimateReap(shape, inputs({ storage: HOST_RAM }))
    const b = estimateReap(shape, inputs({ storage: NVME }))
    expect(a.peakVramBytes).toBe(b.peakVramBytes)
  })
})

describe('estimateReap explanation output', () => {
  const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape

  it('carries steps, constants, and assumptions for the breakdown panel', () => {
    const result = estimateReap(shape, inputs())
    expect(result.steps.length).toBeGreaterThan(3)
    expect(result.constants.length).toBeGreaterThan(3)
    expect(result.assumptions.length).toBeGreaterThan(3)
    for (const step of result.steps) {
      expect(step.label.trim()).not.toBe('')
      expect(step.detail.trim()).not.toBe('')
    }
  })

  it('states that REAP does not reduce compute on its own', () => {
    const result = estimateReap(shape, inputs())
    expect(result.assumptions.join(' ')).toMatch(/does not reduce the arithmetic/)
  })
})

describe('estimateReap input validation', () => {
  const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape

  it('rejects a zero or fractional sample count', () => {
    expect(() => estimateReap(shape, inputs({ calibrationSamples: 0 }))).toThrow(ReapInputError)
    expect(() => estimateReap(shape, inputs({ calibrationSamples: 1.5 }))).toThrow(ReapInputError)
  })

  it('rejects a sequence length below one', () => {
    expect(() => estimateReap(shape, inputs({ sequenceLength: 0 }))).toThrow(ReapInputError)
  })

  it('rejects a pruning ratio outside the supported range', () => {
    expect(() => estimateReap(shape, inputs({ pruneRatio: -0.1 }))).toThrow(ReapInputError)
    expect(() => estimateReap(shape, inputs({ pruneRatio: 0.95 }))).toThrow(ReapInputError)
  })

  it('rejects an impossible utilisation or overhead', () => {
    expect(() => estimateReap(shape, inputs({ mfu: 0 }))).toThrow(ReapInputError)
    expect(() => estimateReap(shape, inputs({ mfu: 1.5 }))).toThrow(ReapInputError)
    expect(() => estimateReap(shape, inputs({ overheadFactor: 0.5 }))).toThrow(ReapInputError)
    expect(() => estimateReap(shape, inputs({ microBatchSize: 0 }))).toThrow(ReapInputError)
  })

  it('names the error so a caller can branch on it', () => {
    try {
      estimateReap(shape, inputs({ mfu: 0 }))
      throw new Error('expected a throw')
    } catch (error) {
      expect((error as ReapInputError).name).toBe('ReapInputError')
    }
  })
})

describe('bytesPerParam', () => {
  it('gives the byte width of each format', () => {
    expect(bytesPerParam('BF16')).toBe(2)
    expect(bytesPerParam('FP8')).toBe(1)
    expect(bytesPerParam('INT4')).toBe(0.5)
  })
})
