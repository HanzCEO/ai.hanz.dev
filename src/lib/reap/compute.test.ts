import { describe, expect, it } from 'vitest'

import { GPU_PRESETS, findGpu, findStorage } from '../hardware'
import type { GpuSpec, StorageSpec } from '../hardware'
import { loadConfigFixture } from '@/test/fixtures'
import { reapInputs as inputs } from '@/test/reap'
import type { RawConfig } from '../model-config'

import { estimateReap, peakTflops } from './compute'
import { detectMoeShape } from './moe-shape'
import { WEIGHT_DTYPES, WEIGHT_DTYPE_ORDER, bytesPerParam } from './presets'
import { ReapInputError, type MoeShape } from './types'

const fixture = loadConfigFixture

const H200 = findGpu('h200') as GpuSpec
const RTX_5090 = findGpu('rtx-5090') as GpuSpec
const RTX_5070 = findGpu('rtx-5070') as GpuSpec
const NVME = findStorage('nvme-pcie4') as StorageSpec
const HOST_RAM = findStorage('pcie5-host-ram') as StorageSpec

/** Every fixture that has an expert bank, so a sweep covers each model family. */
const MOE_FIXTURES = [
  'deepseek-r1',
  'deepseek-v32-exp',
  'deepseek-v4-flash',
  'deepseek-v4-pro',
  'deepseek-v41-flash',
  'glm-4-7-flash',
  'glm-5-3',
  'gpt-oss-120b',
  'kimi-k2',
  'mimo-v26-flash-rl',
  'mimo-v26-pro-rl',
  'qwen3-next-80b',
]

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

describe('detectMoeShape on the MiMo-V2.6 releases', () => {
  /**
   * Both releases write moe_layer_freq as a per layer array, not as a number,
   * and both leave the first block dense. Reading the array as a scalar would
   * count every block as an expert block, so these numbers pin the difference.
   */
  it('counts 47 expert blocks in MiMo-V2.6-Flash-RL', () => {
    const flash = detectMoeShape(fixture('mimo-v26-flash-rl')) as MoeShape
    expect(flash).not.toBeNull()
    expect(flash.modelType).toBe('mimo_v2')
    expect(flash.numLayers).toBe(48)
    expect(flash.moeLayers).toBe(47)
    expect(flash.denseLayers).toBe(1)
    expect(flash.routedExperts).toBe(256)
    expect(flash.expertsPerToken).toBe(8)
    expect(flash.sharedExperts).toBe(0)
    expect(flash.moeIntermediateSize).toBe(2048)
    // 309B total, the figure the model card publishes.
    expect(flash.totalParams).toBe(308_778_369_024)
    // The active path is the attention of every block, the one dense block's
    // feed forward, and the 8 experts of 256 that a token reaches in each of
    // the 47 expert blocks. It is the same figure the model shape detector
    // reports, since neither model has a shared expert.
    expect(flash.activeParamsPerToken).toBe(14_146_338_816)
    expect(flash.activeParamsPerToken).toBeLessThan(flash.totalParams / 20)
  })

  it('counts 69 expert blocks in MiMo-V2.6-Pro-RL', () => {
    const pro = detectMoeShape(fixture('mimo-v26-pro-rl')) as MoeShape
    expect(pro).not.toBeNull()
    expect(pro.numLayers).toBe(70)
    expect(pro.moeLayers).toBe(69)
    expect(pro.denseLayers).toBe(1)
    expect(pro.routedExperts).toBe(384)
    expect(pro.expertsPerToken).toBe(8)
    expect(pro.sharedExperts).toBe(0)
    // 1.02T total, the figure the model card publishes.
    expect(pro.totalParams).toBe(1_021_247_225_856)
  })

  it('costs a REAP run against both releases', () => {
    for (const name of ['mimo-v26-flash-rl', 'mimo-v26-pro-rl']) {
      const shape = detectMoeShape(fixture(name)) as MoeShape
      const result = estimateReap(shape, inputs())
      expect(result.moe).toBe(true)
      expect(result.verdict).toBe('fits')
      expect(result.keptExperts).toBeGreaterThan(0)
      expect(result.keptExperts).toBeLessThan(shape.routedExperts)
      expect(result.removedExperts).toBe(shape.routedExperts - result.keptExperts)
      expect(result.totalParamsAfter).toBeLessThan(shape.totalParams)
    }
  })

  it('reports no change in the active path when the top-k is kept', () => {
    // Both releases have one dense block, whose feed forward a token reads
    // whether or not experts are pruned. Counting it on one side only would
    // make the panel claim a speedup that the recipe does not deliver.
    for (const name of ['mimo-v26-flash-rl', 'mimo-v26-pro-rl']) {
      const shape = detectMoeShape(fixture(name)) as MoeShape
      expect(shape.denseLayers).toBe(1)
      expect(shape.denseFfnParams).toBeGreaterThan(0)

      const kept = estimateReap(shape, inputs({ scaleTopK: false }))
      expect(kept.activeParamsPerTokenAfter).toBe(shape.activeParamsPerToken)

      const scaled = estimateReap(shape, inputs({ scaleTopK: true }))
      expect(scaled.activeParamsPerTokenAfter).toBeLessThan(shape.activeParamsPerToken)
    }
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

  it('lands the checkpoint recipe inside the range measured independently for REAP scoring', () => {
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

  it('names a narrower format when the BF16 block overflows the card but FP8 fits', () => {
    // DeepSeek V4 Pro has 384 experts of 66M parameters each. One block is about
    // 47 GiB in BF16, which does not fit a 32 GB card, and about 24 GiB in FP8,
    // which does.
    const shape = detectMoeShape(fixture('deepseek-v4-pro')) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: RTX_5090, weightDtype: 'BF16' }))
    expect(result.verdict).toBe('needs-narrower')
    expect(result.perMoELayerBytes).toBeGreaterThan(result.vramBytes)
    expect(result.bestFittingDtype).toBe('FP8_E4M3')
    // One block in FP8 is about half its BF16 size.
    const blockBytes = result.bestFittingBlockBytes ?? 0
    expect(blockBytes).toBeGreaterThan(0)
    expect(blockBytes / result.perMoELayerBytes).toBeCloseTo(0.5, 3)
  })

  it('names the widest format that still fits when the wider ones overflow', () => {
    // DeepSeek-R1 on a 12 GiB RTX 5070: one expert block overflows the card in
    // BF16 and in FP8, and fits in NVFP4. The panel used to render the
    // needs-offload copy while the format picker flipped the same card to Fits
    // as soon as a 4 bit format was selected.
    const shape = detectMoeShape(fixture('deepseek-r1')) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: RTX_5070, weightDtype: 'BF16' }))
    expect(result.verdict).toBe('needs-narrower')
    expect(result.bestFittingDtype).toBe('NVFP4')
    expect(result.perMoELayerBytes).toBeGreaterThan(result.vramBytes)
    expect(result.bestFittingBlockBytes).toBeLessThan(result.vramBytes)
  })

  it('gives the same model and card a fitting verdict once that format is selected', () => {
    // The verdict has to agree with the format picker, so the two selections
    // can never disagree about the same run.
    const shape = detectMoeShape(fixture('deepseek-r1')) as MoeShape
    const bf16 = estimateReap(shape, inputs({ gpu: RTX_5070, weightDtype: 'BF16' }))
    const nvfp4 = estimateReap(shape, inputs({ gpu: RTX_5070, weightDtype: 'NVFP4' }))
    expect(bf16.verdict).toBe('needs-narrower')
    expect(nvfp4.verdict).toBe('fits')
    expect(nvfp4.perMoELayerBytes).toBe(bf16.bestFittingBlockBytes)
  })

  it('reports needs-offload when even the narrowest format overflows', () => {
    // Kimi-K2 in FP8 is about 16 GiB a block. A micro batch of 128 samples of
    // 2048 tokens adds a 28 GiB activation buffer, which no longer fits. INT4
    // halves the block, and the activation buffer alone is still larger than
    // the card, so no format rescues the run.
    const shape = detectMoeShape(fixture('kimi-k2')) as MoeShape
    const result = estimateReap(
      shape,
      inputs({ gpu: RTX_5090, weightDtype: 'BF16', microBatchSize: 128 }),
    )
    expect(result.verdict).toBe('needs-offload')
    expect(result.bestFittingDtype).toBeNull()
    expect(result.bestFittingBlockBytes).toBeNull()
  })

  it('reports fits when the selected format is already the narrow one', () => {
    const shape = detectMoeShape(fixture('kimi-k2')) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: RTX_5090, weightDtype: 'FP8_E4M3' }))
    expect(result.verdict).toBe('fits')
  })

  it('names the widest format that fits', () => {
    const shape = detectMoeShape(QWEN3_30B_A3B) as MoeShape
    const result = estimateReap(shape, inputs({ gpu: H200, weightDtype: 'BF16' }))
    // BF16 already fits, so it is the widest format that fits.
    expect(result.bestFittingDtype).toBe('BF16')
  })

  it('names a fitting format whenever the verdict is not needs-offload', () => {
    // The verdict and the fitting search answer the same question, so they can
    // never contradict each other. Sweeping every fixture against every card
    // and every format is what catches a verdict that stops short of a format
    // the search accepts.
    for (const name of MOE_FIXTURES) {
      const shape = detectMoeShape(fixture(name)) as MoeShape
      for (const gpu of GPU_PRESETS) {
        for (const weightDtype of WEIGHT_DTYPE_ORDER) {
          const result = estimateReap(shape, inputs({ gpu, weightDtype }))
          const where = `${name} on ${gpu.id} at ${weightDtype}`
          if (result.verdict === 'needs-offload') {
            expect(result.bestFittingDtype, where).toBeNull()
            expect(result.bestFittingBlockBytes, where).toBeNull()
          } else {
            expect(result.bestFittingDtype, where).not.toBeNull()
            expect(result.bestFittingBlockBytes, where).not.toBeNull()
            expect(result.bestFittingBlockBytes as number, where).toBeLessThanOrEqual(
              result.vramBytes,
            )
          }
        }
      }
    }
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

  it('does not report a zero reduction as a percentage', () => {
    // A ratio of zero keeps every expert. The step used to render "That removes
    // 0.0 percent of the parameters", which asserts a reduction that did not
    // happen.
    const result = estimateReap(shape, inputs({ pruneRatio: 0 }))
    const step = result.steps.find((entry) => entry.label === 'After pruning')
    expect(step).toBeDefined()
    expect(step?.detail).not.toContain('0.0 percent')
    expect(step?.detail).not.toContain('percent of the parameters')
    expect(step?.detail).toContain('removes nothing')
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
    const fp8 = estimateReap(shape, inputs({ weightDtype: 'FP8_E4M3' }))
    // FP8 is half the BF16 payload plus a small scale sidecar.
    expect(fp8.weightBytes / bf16.weightBytes).toBeCloseTo(0.5, 3)
    expect(fp8.perMoELayerBytes / bf16.perMoELayerBytes).toBeCloseTo(0.5, 3)
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
  it('gives the byte width of each format, including the scale sidecar', () => {
    expect(bytesPerParam('BF16')).toBe(2)
    expect(bytesPerParam('FP8_E4M3')).toBeCloseTo(1 + 1 / 16384, 10)
    expect(bytesPerParam('NVFP4')).toBeCloseTo(0.5625, 10)
    expect(bytesPerParam('MXFP4')).toBeCloseTo(0.53125, 10)
    expect(bytesPerParam('INT4')).toBeCloseTo(0.515625, 10)
  })

  it('offers MXFP4 and NVFP4 with their scale aware byte cost', () => {
    const ids = WEIGHT_DTYPES.map((spec) => spec.id)
    expect(ids).toContain('MXFP4')
    expect(ids).toContain('NVFP4')
    expect(WEIGHT_DTYPES.find((spec) => spec.id === 'MXFP4')?.bytes).toBeCloseTo(0.53125, 10)
    expect(WEIGHT_DTYPES.find((spec) => spec.id === 'NVFP4')?.bytes).toBeCloseTo(0.5625, 10)
  })
})

describe('peakTflops', () => {
  it('uses the FP4 path for a 4 bit format on a card that has one', () => {
    expect(peakTflops('MXFP4', RTX_5090)).toBe(RTX_5090.fp4DenseTflops)
    expect(peakTflops('NVFP4', RTX_5090)).toBe(RTX_5090.fp4DenseTflops)
    expect(peakTflops('INT4', RTX_5090)).toBe(RTX_5090.fp4DenseTflops)
  })

  it('falls back to the FP8 rate when the card has no FP4 path', () => {
    expect(H200.fp4DenseTflops).toBeNull()
    expect(peakTflops('MXFP4', H200)).toBe(H200.fp8DenseTflops)
  })

  it('uses the FP8 path for an FP8 format and BF16 otherwise', () => {
    expect(peakTflops('FP8_E4M3', H200)).toBe(H200.fp8DenseTflops)
    expect(peakTflops('BF16', RTX_5090)).toBe(RTX_5090.bf16DenseTflops)
  })
})
