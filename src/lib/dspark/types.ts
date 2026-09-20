import type { GpuSpec, StorageSpec } from '../hardware'

/**
 * How the target's supervision reaches the draft during training.
 *
 * Offline precomputes the target's hidden states once and writes them to disk,
 * so the target never has to be resident while the draft trains. It costs a very
 * large amount of storage and makes the run I/O bound. Online captures the
 * hidden states on the fly, so nothing is written, but the target has to stay
 * loaded for the whole run.
 */
export type DsparkDataMode = 'offline' | 'online'

/**
 * The shape a target config describes, in the terms this cost model needs.
 *
 * DSpark trains a small drafter against a frozen target, so the target's width
 * and depth set both the cache width and the amount of context each draft block
 * has to attend to. Nothing here is specific to a model family.
 */
export interface DsparkTargetShape {
  modelType: string
  hiddenSize: number
  /** Transformer blocks in the target. */
  numLayers: number
  /** Dense feed forward width. */
  intermediateSize: number
  vocabSize: number
  /** Parameters in one attention block of the target. */
  attentionParamsPerLayer: number

  /** Routed experts per MoE block, zero on a dense target. */
  routedExperts: number
  /** Experts one token routes to. */
  expertsPerToken: number
  /** Expert feed forward width. */
  moeIntermediateSize: number
  /** Blocks that contain an expert bank. */
  moeLayers: number

  /** Parameters that exist in the target file. */
  totalParams: number
  /** Parameters the target touches per token, router excluded. */
  activeParamsPerToken: number

  /** True when a field had to be inferred rather than read. */
  bestEffort: boolean
  /**
   * True when the config carries DSpark draft fields, which means it may be a
   * draft checkpoint rather than the target. Its num_hidden_layers would then
   * be the draft's depth, and every figure derived from the depth is wrong.
   */
  looksLikeDraftConfig: boolean
  notes: string[]
}

export interface DsparkInputs {
  /** Captured target layers. Each one adds a hidden state per token to the cache. */
  numTargetLayers: number
  /** Tokens in the training set, one pass. The dominant cost driver. */
  trainingTokens: number
  /** Passes over the training set. */
  epochs: number
  /** Blocks sampled from each sequence per step. */
  numAnchors: number
  /** Tokens drafted per block, the gamma of the paper. */
  blockSize: number
  /** Draft backbone depth. */
  numDraftLayers: number
  /** Rank of the low-rank Markov head. */
  markovRank: number
  /** Packed sequence length the cache is built at. */
  sequenceLength: number
  dataMode: DsparkDataMode
  gpu: GpuSpec
  storage: StorageSpec
  /** Cards the run is spread over. */
  gpuCount: number
  /** Model flops utilisation, between 0.01 and 1. */
  mfu: number
  /** Multiplier over the pure compute or stream time. */
  overheadFactor: number
  /** Fixed cost in seconds: load, prepare the cache, write the checkpoint. */
  setupSeconds: number
  /** Sequences held in flight at once. */
  microBatchSize: number
}

export type DsparkVerdict =
  /** The run fits the cards it is configured for. */
  | 'fits'
  /** Online capture does not fit but the offline cache does. */
  | 'needs-offline'
  /** One card is not enough, but the run shards onto a few. */
  | 'needs-more-gpus'
  /** No reasonable number of cards holds it. */
  | 'needs-offload'

/** Whether the run is limited by the GPU or by reading the cache back. */
export type DsparkBound = 'compute' | 'cache-read'

export interface DsparkResult {
  mode: DsparkDataMode
  verdict: DsparkVerdict
  shape: DsparkTargetShape

  // --- The target cache, which only the offline mode writes ---------------
  cacheBytesPerToken: number
  cacheBytes: number
  cacheWriteSeconds: number
  cacheReadSeconds: number

  // --- The draft model ----------------------------------------------------
  draftBackboneParams: number
  draftProjectionParams: number
  draftMarkovParams: number
  draftConfidenceParams: number
  draftParams: number
  draftWeightBytes: number
  optimizerBytes: number
  gradientBytes: number

  // --- Arithmetic ---------------------------------------------------------
  /**
   * Tokens in one pass over the training set. Repeated here because online mode
   * reports a cache of zero, so the panel cannot recover the token count from
   * the cache size.
   */
  trainingTokens: number
  /** Captured target layers, the recipe field that sets the cache width. */
  numTargetLayers: number
  /**
   * Anchors actually used per sequence. Lower than the input when the requested
   * count would score more positions than the sequence holds.
   */
  numAnchors: number
  /** True when the anchor count had to be reduced to fit the sequence. */
  anchorsClamped: boolean
  /** Tokens drafted per block, the gamma the drafter proposes at inference. */
  blockSize: number
  sequencesPerEpoch: number
  positionsPerEpoch: number
  trainingFlops: number
  contextFlops: number
  cachePrepFlops: number
  totalFlops: number
  computeSeconds: number

  // --- Time ---------------------------------------------------------------
  bound: DsparkBound
  estimateSeconds: number
  setupSeconds: number
  overheadFactor: number

  // --- Memory -------------------------------------------------------------
  /** Sequences in flight on each card, which is what sizes the buffer. */
  activationBytes: number
  targetWeightBytes: number
  runtimeReserveBytes: number
  /** Cards the run is spread over, as entered. */
  gpuCount: number
  /**
   * Weights, optimizer state, gradients and the resident target, added up over
   * the whole run. A run on several cards holds a slice of this, not a copy.
   */
  modelStateBytes: number
  /** The share of the model state that one card holds. */
  perCardStateBytes: number
  /**
   * The peak one card holds: its share of the model state, plus the activation
   * buffer and the framework reserve, which do not divide. This is the figure
   * the verdict compares against the VRAM of one card.
   */
  peakVramBytes: number
  vramBytes: number
  /**
   * The smallest card count whose share of the model state fits beside the
   * per-card terms. Null when those terms alone exceed one card, which no card
   * count can fix.
   */
  gpusNeeded: number | null

  steps: Array<{ label: string; detail: string }>
  constants: Array<{ key: string; value: number | string; source: string }>
  assumptions: string[]
}

/** The estimator input a validation error belongs to, when it has one. */
export type DsparkInputField =
  | 'trainingTokens'
  | 'epochs'
  | 'numAnchors'
  | 'blockSize'
  | 'numDraftLayers'
  | 'numTargetLayers'
  | 'markovRank'
  | 'sequenceLength'
  | 'gpuCount'
  | 'mfu'
  | 'overheadFactor'
  | 'microBatchSize'

export class DsparkInputError extends Error {
  /** The field to mark invalid in the form, or null when no single field owns it. */
  readonly field: DsparkInputField | null

  constructor(message: string, field: DsparkInputField | null = null) {
    super(message)
    this.name = 'DsparkInputError'
    this.field = field
  }
}
