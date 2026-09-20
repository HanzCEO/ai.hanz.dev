import { describe, expect, it } from 'vitest'

import { findGpu, findStorage } from '../hardware'
import type { GpuSpec, StorageSpec } from '../hardware'
import type { RawConfig } from '../model-config'

import { estimateDspark } from './compute'
import {
  ACTIVATION_FACTOR,
  BF16_BYTES,
  DEFAULT_BLOCK_SIZE,
  DEFAULT_DRAFT_LAYERS,
  DEFAULT_MARKOV_RANK,
  DEFAULT_MFU,
  DEFAULT_MICRO_BATCH,
  DEFAULT_NUM_ANCHORS,
  DEFAULT_OVERHEAD_FACTOR,
  DEFAULT_SEQUENCE_LENGTH,
  DEFAULT_SETUP_SECONDS,
  DEFAULT_TARGET_LAYERS,
  RUNTIME_OVERHEAD_BYTES,
} from './presets'
import { detectDsparkShape } from './shape'
import {
  DsparkInputError,
  type DsparkInputs,
  type DsparkTargetShape,
  type DsparkVerdict,
} from './types'

/** A card chosen so the arithmetic below is exact and checkable by hand. */
const TEST_GPU: GpuSpec = {
  id: 'test',
  label: 'Test card',
  vendor: 'nvidia',
  generation: 'test',
  vramGiB: 2,
  bandwidthGBs: 1000,
  bf16DenseTflops: 1,
  fp8DenseTflops: null,
  note: 'A fixture, not a real part.',
}

const TEST_STORAGE: StorageSpec = {
  id: 'test',
  label: 'Test storage',
  bandwidthGBs: 1000,
  note: 'A fixture, not a real device.',
}

/**
 * A target small enough that every intermediate figure can be worked out by
 * hand, which is what the exact-value test below does.
 *
 * hidden 8, 2 layers, dense width 16, vocab 10, 2 heads of dimension 4 with one
 * key value head. Attention per layer works out at 8*2*4 + 2*8*1*4 + 2*4*8,
 * which is 192.
 */
const TINY_TARGET: RawConfig = {
  model_type: 'tiny',
  hidden_size: 8,
  num_hidden_layers: 2,
  intermediate_size: 16,
  vocab_size: 10,
  num_attention_heads: 2,
  num_key_value_heads: 1,
  head_dim: 4,
}

/**
 * Qwen3-4B, which is the target the DeepSpec README quotes its storage figure
 * against. The shape is what makes the 38 TB anchor reproducible.
 */
const QWEN3_4B: RawConfig = {
  model_type: 'qwen3',
  hidden_size: 2560,
  num_hidden_layers: 36,
  intermediate_size: 9728,
  vocab_size: 151936,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  head_dim: 128,
  tie_word_embeddings: false,
}

function shapeFrom(config: RawConfig): DsparkTargetShape {
  const shape = detectDsparkShape(config)
  if (!shape) throw new Error('the fixture config should have been readable')
  return shape
}

const TINY = shapeFrom(TINY_TARGET)
const QWEN = shapeFrom(QWEN3_4B)

function inputs(overrides: Partial<DsparkInputs> = {}): DsparkInputs {
  return {
    numTargetLayers: DEFAULT_TARGET_LAYERS,
    trainingTokens: 1_237_000_000,
    epochs: 10,
    numAnchors: DEFAULT_NUM_ANCHORS,
    blockSize: DEFAULT_BLOCK_SIZE,
    numDraftLayers: DEFAULT_DRAFT_LAYERS,
    markovRank: DEFAULT_MARKOV_RANK,
    sequenceLength: DEFAULT_SEQUENCE_LENGTH,
    dataMode: 'offline',
    // The reference configurations assume a single node of eight, and the
    // published recipe quotes its storage figure against exactly that.
    gpu: findGpu('h200') as GpuSpec,
    storage: findStorage('pcie5-host-ram') as StorageSpec,
    gpuCount: 8,
    mfu: DEFAULT_MFU,
    overheadFactor: DEFAULT_OVERHEAD_FACTOR,
    setupSeconds: DEFAULT_SETUP_SECONDS,
    microBatchSize: DEFAULT_MICRO_BATCH,
    ...overrides,
  }
}

/** The inputs for the hand worked example below, spelled out in full. */
function tinyInputs(overrides: Partial<DsparkInputs> = {}): DsparkInputs {
  return inputs({
    numTargetLayers: 2,
    trainingTokens: 64,
    epochs: 1,
    numAnchors: 4,
    blockSize: 2,
    numDraftLayers: 2,
    markovRank: 2,
    sequenceLength: 8,
    gpu: TEST_GPU,
    storage: TEST_STORAGE,
    gpuCount: 1,
    mfu: 0.5,
    overheadFactor: 1,
    setupSeconds: 0,
    microBatchSize: 1,
    ...overrides,
  })
}

describe('detectDsparkShape', () => {
  it('reads a dense target', () => {
    expect(TINY.modelType).toBe('tiny')
    expect(TINY.hiddenSize).toBe(8)
    expect(TINY.numLayers).toBe(2)
    expect(TINY.attentionParamsPerLayer).toBe(192)
    expect(TINY.moeLayers).toBe(0)
    expect(TINY.bestEffort).toBe(false)
  })

  it('counts the parameters of a real target', () => {
    // 36 layers of attention at 26214400 is 943718400, 36 dense blocks at
    // 3*2560*9728 is 2689597440, and the embedding at 151936*2560 is
    // 388956160. Qwen3-4B is a 4.02 billion parameter model.
    expect(QWEN.attentionParamsPerLayer).toBe(26_214_400)
    expect(QWEN.totalParams).toBe(4_022_272_000)
    // Active parameters are the ones a matmul touches, so they exclude the
    // embedding, which is a lookup. On a dense target that is everything else.
    expect(QWEN.activeParamsPerToken).toBe(3_633_315_840)
    expect(QWEN.totalParams - QWEN.activeParamsPerToken).toBe(388_956_160)
  })

  it('returns null for a config with no transformer shape', () => {
    expect(detectDsparkShape({ model_type: 'clip' })).toBeNull()
    expect(detectDsparkShape({ hidden_size: 4096 })).toBeNull()
  })

  it('reads an expert bank through the shared reader', () => {
    const moe = shapeFrom({
      model_type: 'qwen3_moe',
      hidden_size: 2048,
      num_hidden_layers: 48,
      intermediate_size: 6144,
      moe_intermediate_size: 768,
      n_routed_experts: 128,
      num_experts_per_tok: 8,
      vocab_size: 151936,
      num_attention_heads: 32,
      num_key_value_heads: 4,
      head_dim: 128,
    })
    expect(moe.moeLayers).toBe(48)
    expect(moe.routedExperts).toBe(128)
    // Only the routed experts a token reaches count toward active parameters.
    expect(moe.activeParamsPerToken).toBeLessThan(moe.totalParams / 8)
  })
})

/**
 * Every figure in this block is worked out by hand from the tiny fixture, so a
 * change to any term in the cost model shows up here rather than hiding behind
 * a plausible looking final number.
 */
describe('estimateDspark, a hand computed case', () => {
  const result = estimateDspark(TINY, tinyInputs())

  it('sizes the cache row', () => {
    // 2 layers * 8 values * 2 bytes = 32, the last hidden state is 8 * 2 = 16,
    // the token ids are 4, and the two masks are 1 each.
    expect(result.cacheBytesPerToken).toBe(32 + 16 + 4 + 1 + 1)
    expect(result.cacheBytesPerToken).toBe(54)
    expect(result.cacheBytes).toBe(54 * 64)
  })

  it('counts the draft parameters', () => {
    // backbone: 2 * (192 attention + 3*8*16 ffn) = 2 * 576
    expect(result.draftBackboneParams).toBe(1152)
    // projection: 2 captured layers * 8 * 8
    expect(result.draftProjectionParams).toBe(128)
    // markov: 2 * vocab 10 * rank 2
    expect(result.draftMarkovParams).toBe(40)
    // confidence: hidden + rank
    expect(result.draftConfidenceParams).toBe(10)
    expect(result.draftParams).toBe(1152 + 128 + 40 + 10)
    expect(result.draftParams).toBe(1330)
  })

  it('counts the positions scored', () => {
    // 64 tokens at 8 per sequence is 8 sequences, each giving 4 anchors of 2.
    expect(result.sequencesPerEpoch).toBe(8)
    expect(result.positionsPerEpoch).toBe(64)
  })

  it('counts the arithmetic', () => {
    // 6 flops per parameter per position, one epoch.
    expect(result.trainingFlops).toBe(6 * 64 * 1 * 1330)
    expect(result.trainingFlops).toBe(510_720)
    // context: 4 * anchors 4 * block 2 * seq 8 * hidden 8 * layers 2, per sequence
    expect(result.contextFlops).toBe(4 * 4 * 2 * 8 * 8 * 2 * 8)
    expect(result.contextFlops).toBe(32_768)
    // cache prep: a forward pass, so 2 flops per target parameter per token.
    expect(result.cachePrepFlops).toBe(2 * 64 * 1152)
    expect(result.cachePrepFlops).toBe(147_456)
    expect(result.totalFlops).toBe(510_720 + 32_768 + 147_456)
    expect(result.totalFlops).toBe(690_944)
  })

  it('divides the arithmetic by the card', () => {
    // 1e12 flops per second at 1 TFLOPS, halved by the utilisation.
    expect(result.computeSeconds).toBeCloseTo(690_944 / (1e12 * 0.5), 12)
  })

  it('counts the memory', () => {
    expect(result.draftWeightBytes).toBe(1330 * 2)
    expect(result.optimizerBytes).toBe(1330 * 8)
    expect(result.gradientBytes).toBe(1330 * 2)
    // micro batch 1 * 4 anchors * 2 block * 8 hidden * 2 bytes * 6 intermediates
    expect(result.activationBytes).toBe(1 * 4 * 2 * 8 * BF16_BYTES * ACTIVATION_FACTOR)
    expect(result.activationBytes).toBe(768)
    // Offline, so the target is not resident while the draft trains.
    expect(result.targetWeightBytes).toBe(0)
    expect(result.peakVramBytes).toBe(
      2660 + 10_640 + 2660 + 768 + 0 + RUNTIME_OVERHEAD_BYTES,
    )
  })

  it('reports a self consistent set of steps', () => {
    expect(result.steps).toHaveLength(9)
    expect(result.assumptions.length).toBeGreaterThan(5)
    expect(result.constants.some((constant) => constant.key === 'draft_params')).toBe(true)
  })
})

describe('the published storage anchor', () => {
  /**
   * The DeepSpec README states that the default Qwen3-4B setting takes roughly
   * 38 TB of target cache. That setting is one pass over Open-PerfectBlend,
   * which is 1.3 million samples, at a packed sequence length of 4096, and the
   * quoted figure is decimal terabytes. Reproducing it is the check that the
   * cache row is laid out the way the implementation lays it out.
   */
  it('reproduces the 38 TB the DeepSpec README quotes', () => {
    const result = estimateDspark(QWEN, inputs())
    const quoted = 38e12
    const error = Math.abs(result.cacheBytes - quoted) / quoted
    expect(error).toBeLessThan(0.05)
    // 1.237 billion tokens at 30726 bytes each.
    expect(result.cacheBytesPerToken).toBe(30_726)
  })

  it('scales the cache with the captured layer count', () => {
    const five = estimateDspark(QWEN, inputs({ numTargetLayers: 5 }))
    const two = estimateDspark(QWEN, inputs({ numTargetLayers: 2 }))
    // The captured layers alone would give 2.5. The last hidden state, the
    // token ids and the masks do not scale with the layer count, so the real
    // ratio lands just under 2.
    expect(five.cacheBytesPerToken).toBe(5 * 5120 + 5120 + 6)
    expect(two.cacheBytesPerToken).toBe(2 * 5120 + 5120 + 6)
    expect(five.cacheBytes / two.cacheBytes).toBeCloseTo(30_726 / 15_366, 6)
    expect(five.cacheBytes / two.cacheBytes).toBeLessThan(2)
  })
})

describe('what bounds the run', () => {
  const hostRam = findStorage('pcie5-host-ram') as StorageSpec
  const network = findStorage('network-10gbe') as StorageSpec

  it('is compute when the cache sits in host memory', () => {
    const result = estimateDspark(QWEN, inputs({ storage: hostRam }))
    expect(result.bound).toBe('compute')
    expect(result.computeSeconds).toBeGreaterThan(result.cacheReadSeconds)
  })

  it('becomes the cache read over a network share', () => {
    const result = estimateDspark(QWEN, inputs({ storage: network }))
    expect(result.bound).toBe('cache-read')
    expect(result.cacheReadSeconds).toBeGreaterThan(result.computeSeconds)
    // The estimate follows the slower of the two.
    expect(result.estimateSeconds).toBeGreaterThan(
      result.computeSeconds * DEFAULT_OVERHEAD_FACTOR,
    )
  })

  it('reads the cache back once per epoch', () => {
    const one = estimateDspark(QWEN, inputs({ storage: network, epochs: 1 }))
    const two = estimateDspark(QWEN, inputs({ storage: network, epochs: 2 }))
    expect(two.cacheReadSeconds).toBeCloseTo(one.cacheReadSeconds * 2, 6)
    // The cache is written once, not once per epoch.
    expect(two.cacheWriteSeconds).toBeCloseTo(one.cacheWriteSeconds, 6)
  })
})

describe('offline against online', () => {
  const hostRam = findStorage('pcie5-host-ram') as StorageSpec

  it('writes a cache offline and nothing online', () => {
    const offline = estimateDspark(QWEN, inputs({ dataMode: 'offline', storage: hostRam }))
    const online = estimateDspark(QWEN, inputs({ dataMode: 'online', storage: hostRam }))

    expect(offline.cacheBytes).toBeGreaterThan(0)
    expect(online.cacheBytes).toBe(0)
    expect(online.cacheWriteSeconds).toBe(0)
    expect(online.cacheReadSeconds).toBe(0)
    // With nothing to read, the run can only be bound by compute.
    expect(online.bound).toBe('compute')
    // Online also skips the target forward pass that builds the cache.
    expect(online.cachePrepFlops).toBe(0)
    expect(online.totalFlops).toBeLessThan(offline.totalFlops)
  })

  it('keeps the target resident online and not offline', () => {
    const offline = estimateDspark(QWEN, inputs({ dataMode: 'offline', storage: hostRam }))
    const online = estimateDspark(QWEN, inputs({ dataMode: 'online', storage: hostRam }))
    expect(offline.targetWeightBytes).toBe(0)
    // Qwen3-4B in bf16 is about 7.5 GiB of weights.
    expect(online.targetWeightBytes).toBe(QWEN.totalParams * 2)
    expect(online.peakVramBytes - offline.peakVramBytes).toBe(online.targetWeightBytes)
  })
})

describe('the memory verdict', () => {
  const hostRam = findStorage('pcie5-host-ram') as StorageSpec

  function verdictFor(gpuId: string, mode: 'offline' | 'online'): DsparkVerdict {
    return estimateDspark(
      QWEN,
      inputs({ gpu: findGpu(gpuId) as GpuSpec, dataMode: mode, storage: hostRam }),
    ).verdict
  }

  it('fits a large card either way', () => {
    // 32 GiB holds the 615 million parameter drafter and its state comfortably,
    // with room for a resident 4B target on top.
    expect(verdictFor('rtx-5090', 'offline')).toBe('fits')
    expect(verdictFor('rtx-5090', 'online')).toBe('fits')
  })

  it('sends a small card offline rather than failing', () => {
    // 12 GiB holds the drafter but not the drafter plus a resident target, so
    // the actionable answer is to precompute the cache.
    expect(verdictFor('rtx-5070', 'offline')).toBe('fits')
    expect(verdictFor('rtx-5070', 'online')).toBe('needs-offline')
  })

  it('changes its answer with the card', () => {
    // 8 GiB does not hold the drafter and its optimizer state at all, so the
    // advice becomes more hardware rather than a different data mode.
    expect(verdictFor('rtx-4060', 'offline')).toBe('needs-more-gpus')
    expect(verdictFor('rtx-5090', 'offline')).toBe('fits')
  })

  it('says how many cards the peak needs', () => {
    const result = estimateDspark(
      QWEN,
      inputs({ gpu: findGpu('rtx-4060') as GpuSpec, storage: hostRam }),
    )
    expect(result.gpusNeeded).toBe(2)
    expect(result.gpusNeeded * result.vramBytes).toBeGreaterThanOrEqual(result.peakVramBytes)
  })

  it('shrinks the memory with a smaller micro batch', () => {
    const one = estimateDspark(QWEN, inputs({ microBatchSize: 1, storage: hostRam }))
    const four = estimateDspark(QWEN, inputs({ microBatchSize: 4, storage: hostRam }))
    expect(four.activationBytes).toBe(one.activationBytes * 4)
    expect(four.peakVramBytes).toBeGreaterThan(one.peakVramBytes)
  })
})

describe('validation', () => {
  const cases: Array<[Partial<DsparkInputs>, string]> = [
    [{ epochs: 0 }, 'epochs'],
    [{ trainingTokens: 0 }, 'trainingTokens'],
    [{ numAnchors: 0 }, 'numAnchors'],
    [{ blockSize: 0 }, 'blockSize'],
    [{ numDraftLayers: 0 }, 'numDraftLayers'],
    [{ numTargetLayers: 0 }, 'numTargetLayers'],
    [{ sequenceLength: 0 }, 'sequenceLength'],
    [{ gpuCount: 0 }, 'gpuCount'],
    [{ microBatchSize: 0 }, 'microBatchSize'],
    [{ markovRank: -1 }, 'markovRank'],
    [{ mfu: 0 }, 'mfu'],
    [{ overheadFactor: 0.5 }, 'overheadFactor'],
  ]

  for (const [overrides, field] of cases) {
    it(`names ${field} when it is out of range`, () => {
      try {
        estimateDspark(TINY, tinyInputs(overrides))
        throw new Error('the estimator should have rejected these inputs')
      } catch (error) {
        expect(error).toBeInstanceOf(DsparkInputError)
        expect((error as DsparkInputError).field).toBe(field)
      }
    })
  }

  it('accepts a disabled Markov head', () => {
    const result = estimateDspark(TINY, tinyInputs({ markovRank: 0 }))
    expect(result.draftMarkovParams).toBe(0)
  })

  it('accepts a utilisation of exactly one', () => {
    expect(() => estimateDspark(TINY, tinyInputs({ mfu: 1 }))).not.toThrow()
  })
})

describe('the reported figures stay coherent', () => {
  const result = estimateDspark(QWEN, inputs())

  it('adds the memory terms up to the peak', () => {
    expect(result.peakVramBytes).toBe(
      result.draftWeightBytes +
        result.optimizerBytes +
        result.gradientBytes +
        result.activationBytes +
        result.targetWeightBytes +
        result.runtimeReserveBytes,
    )
  })

  it('adds the arithmetic terms up to the total', () => {
    expect(result.totalFlops).toBe(
      result.trainingFlops + result.contextFlops + result.cachePrepFlops,
    )
  })

  it('keeps the draft far smaller than the target it imitates', () => {
    expect(result.draftParams).toBeLessThan(QWEN.totalParams)
    expect(result.draftParams).toBeGreaterThan(0)
  })

  it('puts the estimate above the bound it reports', () => {
    const boundSeconds =
      result.bound === 'compute' ? result.computeSeconds : result.cacheReadSeconds
    expect(result.estimateSeconds).toBeGreaterThanOrEqual(boundSeconds)
  })
})
