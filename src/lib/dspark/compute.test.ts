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
 * Qwen3-4B, which the DeepSpec storage figure is quoted against. The tie flag
 * is the real one: Qwen3-4B ships a tied embedding, so the language model head
 * is not stored twice. Its published checkpoint holds 4,022,468,096 parameters.
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
  tie_word_embeddings: true,
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
    // The target is the whole of the difference in model state. The peak that
    // one card holds divides that state, so it is the state that is compared.
    expect(online.modelStateBytes - offline.modelStateBytes).toBe(online.targetWeightBytes)
  })
})

describe('the memory verdict', () => {
  const hostRam = findStorage('pcie5-host-ram') as StorageSpec

  function verdictFor(gpuId: string, mode: 'offline' | 'online'): DsparkVerdict {
    // One card, because the verdict is about what a single card can hold.
    return estimateDspark(
      QWEN,
      inputs({ gpu: findGpu(gpuId) as GpuSpec, dataMode: mode, storage: hostRam, gpuCount: 1 }),
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

  it('says how many cards the model state needs', () => {
    const result = estimateDspark(
      QWEN,
      inputs({ gpu: findGpu('rtx-4060') as GpuSpec, storage: hostRam, gpuCount: 1 }),
    )
    const needed = result.gpusNeeded
    expect(needed).not.toBeNull()
    expect(needed).toBe(2)
    // At that count the share of the model state fits beside the per-card terms.
    const peakFor = (cards: number) =>
      result.modelStateBytes / cards + result.activationBytes + result.runtimeReserveBytes
    expect(peakFor(needed as number)).toBeLessThanOrEqual(result.vramBytes)
    // One card fewer does not.
    expect(peakFor((needed as number) - 1)).toBeGreaterThan(result.vramBytes)
  })

  it('shrinks the memory with a smaller micro batch', () => {
    const one = estimateDspark(QWEN, inputs({ microBatchSize: 1, storage: hostRam }))
    const four = estimateDspark(QWEN, inputs({ microBatchSize: 4, storage: hostRam }))
    expect(four.activationBytes).toBe(one.activationBytes * 4)
    expect(four.peakVramBytes).toBeGreaterThan(one.peakVramBytes)
  })
})

/**
 * The card count has to reach the memory verdict, not only the duration. A run
 * that is spread over enough cards fits even when one card does not, and the
 * sentence the panel prints has to name the count the run actually uses.
 */
describe('the card count', () => {
  const hostRam = findStorage('pcie5-host-ram') as StorageSpec
  const small = findGpu('rtx-4060') as GpuSpec

  it('divides the model state and leaves the per card terms alone', () => {
    const one = estimateDspark(QWEN, inputs({ gpuCount: 1, storage: hostRam }))
    const four = estimateDspark(QWEN, inputs({ gpuCount: 4, storage: hostRam }))
    expect(four.modelStateBytes).toBe(one.modelStateBytes)
    expect(four.perCardStateBytes).toBeCloseTo(one.perCardStateBytes / 4, 6)
    // The activation buffer is held on every card, so it does not divide.
    expect(four.activationBytes).toBe(one.activationBytes)
    expect(four.peakVramBytes).toBeLessThan(one.peakVramBytes)
  })

  it('fits on enough cards when one card is not enough', () => {
    const one = estimateDspark(QWEN, inputs({ gpu: small, gpuCount: 1, storage: hostRam }))
    const four = estimateDspark(QWEN, inputs({ gpu: small, gpuCount: 4, storage: hostRam }))
    expect(one.verdict).toBe('needs-more-gpus')
    expect(four.verdict).toBe('fits')
    // The count the state needs does not depend on how many cards were entered.
    expect(four.gpusNeeded).toBe(one.gpusNeeded)
  })

  it('reports no card count when the per card terms already overflow', () => {
    // A micro batch large enough that the activation buffer alone exceeds the
    // card, which is the one case no card count can fix.
    const result = estimateDspark(
      QWEN,
      inputs({ gpu: small, microBatchSize: 200, storage: hostRam }),
    )
    expect(result.gpusNeeded).toBeNull()
    expect(result.verdict).toBe('needs-offload')
  })

  it('names the card count in the arithmetic step', () => {
    const detailFor = (gpuCount: number) =>
      estimateDspark(QWEN, inputs({ gpuCount, storage: hostRam })).steps.find(
        (step) => step.label === 'Arithmetic duration',
      )?.detail ?? ''
    expect(detailFor(1)).toContain('With 1 GPU at')
    expect(detailFor(8)).toContain('With 8 GPUs at')
  })
})

/**
 * openbmb/MiniCPM5-2B, the target the MiniCPM5-2B-DSpark recipe was trained
 * against. A dense Llama-shaped 2B with 42 blocks.
 */
const MINICPM_TARGET: RawConfig = {
  architectures: ['LlamaForCausalLM'],
  model_type: 'llama',
  hidden_size: 2048,
  num_hidden_layers: 42,
  intermediate_size: 6144,
  vocab_size: 130560,
  num_attention_heads: 16,
  num_key_value_heads: 2,
  head_dim: 128,
  tie_word_embeddings: false,
}

/**
 * openbmb/MiniCPM5-2B-DSpark, the draft checkpoint config.
 *
 * This is a draft config, not a target config: num_hidden_layers is the draft's
 * depth while num_target_layers names the target's. It is here because it is the
 * one published DSpark checkpoint that states its own parameter count, which
 * makes it the only direct check on the draft parameter model.
 */
const MINICPM_DRAFT: RawConfig = {
  architectures: ['Qwen3DSparkModel'],
  model_type: 'qwen3',
  hidden_size: 2048,
  num_hidden_layers: 5,
  intermediate_size: 6144,
  vocab_size: 130560,
  num_attention_heads: 16,
  num_key_value_heads: 2,
  head_dim: 128,
  num_target_layers: 42,
  block_size: 7,
  target_layer_ids: [1, 10, 20, 30, 39],
  markov_rank: 256,
  mask_token_id: 75982,
}

const MINICPM = shapeFrom(MINICPM_TARGET)

/**
 * The MiniCPM5-2B-DSpark recipe, as OpenBMB published it.
 *
 * The card gives 1,959,525 sequences and 7,054,154,509 tokens over 6 epochs,
 * which is 600 tokens per sequence per epoch. Anchors are not published, so they
 * are set to one block per sequence token.
 */
function minicpmInputs(overrides: Partial<DsparkInputs> = {}): DsparkInputs {
  return inputs({
    numTargetLayers: 5,
    trainingTokens: 1_959_525 * 600,
    epochs: 6,
    numAnchors: 85,
    blockSize: 7,
    numDraftLayers: 5,
    markovRank: 256,
    sequenceLength: 600,
    ...overrides,
  })
}

describe('the published MiniCPM5-2B-DSpark recipe', () => {
  /**
   * The model card states the draft parameter count outright, which no other
   * published DSpark checkpoint does. It is the only direct check on the draft
   * parameter model rather than on a figure derived from it.
   */
  it('reproduces the published draft parameter count', () => {
    const result = estimateDspark(MINICPM, minicpmInputs())
    const published = 323_776_001
    // The per-layer RMSNorm weights are the whole of the 25,857 gap. Nothing
    // else in the draft is unaccounted for.
    expect(result.draftParams).toBe(323_750_144)
    expect(published - result.draftParams).toBe(25_857)
    expect(Math.abs(result.draftParams - published) / published).toBeLessThan(0.0001)
  })

  it('breaks the draft count down the way the card implies', () => {
    const result = estimateDspark(MINICPM, minicpmInputs())
    // Five draft blocks of GQA attention plus a 3-wide SwiGLU feed forward.
    expect(result.draftBackboneParams).toBe(5 * (9_437_184 + 3 * 2048 * 6144))
    expect(result.draftBackboneParams).toBe(235_929_600)
    // The projection fuses five captured target layers into the draft width.
    expect(result.draftProjectionParams).toBe(5 * 2048 * 2048)
    expect(result.draftProjectionParams).toBe(20_971_520)
    // The rank 256 Markov head is a full vocabulary by rank pair.
    expect(result.draftMarkovParams).toBe(2 * 130560 * 256)
    expect(result.draftMarkovParams).toBe(66_846_720)
    expect(result.draftConfidenceParams).toBe(2048 + 256)
  })

  it('reproduces the published training token count', () => {
    // The card gives 7,054,154,509 tokens over 6 epochs, which is 1,175,692,418
    // per epoch, or 599.99 tokens for each of 1,959,525 sequences.
    const perEpoch = 1_959_525 * 600
    expect(perEpoch).toBe(1_175_715_000)
    expect(Math.abs(perEpoch * 6 - 7_054_154_509) / 7_054_154_509).toBeLessThan(0.0001)
    expect(Math.abs(perEpoch - 1_175_692_418.1666)).toBeLessThan(25_000)
  })

  it('counts the target parameters the way the checkpoint does', () => {
    // An untied model stores the vocabulary twice. MiniCPM5-2B ships
    // 2,516,756,480 parameters, and the 174,080 gap is the RMSNorm weights.
    expect(MINICPM.totalParams).toBe(2_516_582_400)
    expect(2_516_756_480 - MINICPM.totalParams).toBe(174_080)
    expect(Math.abs(MINICPM.totalParams - 2_516_756_480) / 2_516_756_480).toBeLessThan(0.0001)
  })

  it('sizes the cache from the published target layers', () => {
    const result = estimateDspark(MINICPM, minicpmInputs())
    // Five captured layers of 2048 bf16 values, the last hidden state, int32
    // ids and two uint8 masks.
    expect(result.cacheBytesPerToken).toBe(5 * 2048 * 2 + 2048 * 2 + 4 + 1 + 1)
    expect(result.cacheBytesPerToken).toBe(24_582)
    expect(result.cacheBytes).toBe(24_582 * 1_175_715_000)
  })

  it('reads the target as a dense model with no expert bank', () => {
    expect(MINICPM.routedExperts).toBe(0)
    expect(MINICPM.moeLayers).toBe(0)
    expect(MINICPM.numLayers).toBe(42)
    expect(MINICPM.bestEffort).toBe(false)
  })
})

describe('the anchor cap', () => {
  it('caps anchors at one block per sequence token', () => {
    // The DeepSpec default of 512 anchors assumes a long sequence. Pointed at a
    // 600 token sequence it would score more positions than the sequence holds.
    const result = estimateDspark(MINICPM, minicpmInputs({ numAnchors: 512 }))
    expect(result.numAnchors).toBe(85)
    expect(result.anchorsClamped).toBe(true)
    // The cap is what keeps supervision inside the data.
    expect(result.positionsPerEpoch).toBeLessThanOrEqual(result.trainingTokens)
  })

  it('leaves an anchor count that fits alone', () => {
    const result = estimateDspark(MINICPM, minicpmInputs({ numAnchors: 85 }))
    expect(result.numAnchors).toBe(85)
    expect(result.anchorsClamped).toBe(false)
  })

  it('does not cap a recipe that was written for long sequences', () => {
    // 4096 tokens hold 585 blocks of 7, so the published 512 is not touched.
    const result = estimateDspark(QWEN, inputs({ numAnchors: 512, sequenceLength: 4096 }))
    expect(result.numAnchors).toBe(512)
    expect(result.anchorsClamped).toBe(false)
  })

  it('says so in the steps and the assumptions when it caps', () => {
    const result = estimateDspark(MINICPM, minicpmInputs({ numAnchors: 512 }))
    const steps = result.steps.map((step) => step.detail).join(' ')
    expect(steps).toContain('were reduced to 85')
    expect(result.assumptions.some((line) => line.includes('caps the anchor count at 85'))).toBe(true)
  })
})

describe('a draft config read as a target', () => {
  /**
   * A draft checkpoint's own config has a plausible transformer shape, so it
   * reads cleanly and silently. It is flagged instead, because its depth is the
   * draft's and every figure derived from the depth would be wrong.
   */
  it('is flagged rather than silently accepted', () => {
    const shape = detectDsparkShape(MINICPM_DRAFT)
    expect(shape).not.toBeNull()
    expect(shape?.looksLikeDraftConfig).toBe(true)
    // The draft's depth, not the target's 42, which is the whole problem.
    expect(shape?.numLayers).toBe(5)
    expect(shape?.notes.some((note) => note.includes('num_target_layers 42'))).toBe(true)
  })

  it('is not flagged for a plain target config', () => {
    expect(MINICPM.looksLikeDraftConfig).toBe(false)
    expect(QWEN.looksLikeDraftConfig).toBe(false)
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

  it('adds the memory terms up to the peak one card holds', () => {
    expect(result.modelStateBytes).toBe(
      result.draftWeightBytes +
        result.optimizerBytes +
        result.gradientBytes +
        result.targetWeightBytes,
    )
    expect(result.perCardStateBytes).toBe(result.modelStateBytes / result.gpuCount)
    expect(result.peakVramBytes).toBe(
      result.perCardStateBytes + result.activationBytes + result.runtimeReserveBytes,
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
