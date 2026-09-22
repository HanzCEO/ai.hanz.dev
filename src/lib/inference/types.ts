import type { GpuSpec } from '../hardware'
import type { DtypeId } from '../kvcache'
import type { RawConfig } from '../model-config'
import type { ModelShape } from '../model-shape'
import type { WeightFormatId, WeightQuantization } from '../weight-format'

/**
 * The speculative decoding head a model is served with.
 *
 * A head is trained against one model, so the head and the checkpoint are a
 * pair. The multiplier each one carries is held below the published figure,
 * and it is an estimate for measurement rather than a guarantee. Only the
 * decode rate moves with the head. The memory model does not, because a head
 * adds a small number of weights against the whole checkpoint.
 */
export type MtpHeadType = 'none' | 'sequential-mtp' | 'parallel-mtp' | 'medusa' | 'eagle-3'

/** What the recommendation found. */
export type InferenceVerdict =
  /** One card holds the whole run. */
  | 'single'
  /** The run divides across several cards. */
  | 'multi'
  /** No card count up to the limit holds it. */
  | 'none'

export interface InferenceInputs {
  /** The model config, which the KV cache engine reads for the cache shape. */
  config: RawConfig
  /**
   * The weight format to cost the checkpoint in.
   *
   * Absent means the calculator reads the format the checkpoint publishes,
   * which is the default. A value forces that one format on both buckets.
   */
  weightFormat?: WeightFormatId
  /** Tokens in each sequence. */
  contextLength: number
  /** Sequences served at the same time. */
  sequences: number
  /** Share of each card's VRAM held back for fragmentation and the display. */
  headroom: number
  /** The most cards the answer may use. */
  maxGpus: number
  /** Restricts the answer to these GPU ids. Every card is considered when absent. */
  gpuFilter?: string[]
  /**
   * The dtype the KV cache is held in, for example FP8.
   *
   * A cache is far smaller than the weights, but it is the term that grows with
   * the context, so its dtype can decide which card fits. Absent means BF16.
   */
  kvCacheDtype?: DtypeId
  /** The dtype a sparse indexer cache is held in. Absent means BF16. */
  indexerDtype?: DtypeId
  /**
   * The speculative decoding head the model is served with.
   *
   * Absent means no head, which is the bandwidth roofline the calculator
   * already reports. A head raises the decode rate and leaves every memory
   * term alone.
   */
  mtpHead?: MtpHeadType
}

/**
 * One card count of one card model, costed against the model.
 *
 * The weights and the KV cache divide across the tensor-parallel ranks, while
 * the activation buffer and the framework reserve stay in full on every card.
 * That is why a second card does not halve the footprint.
 */
export interface InferenceCandidate {
  gpu: GpuSpec
  /** Cards this configuration uses. */
  gpuCount: number
  /** Bytes one card holds at this card count. */
  perCardBytes: number
  /** Bytes of VRAM one card can use once the headroom is held back. */
  usableBytes: number
  /** VRAM this configuration leaves free on one card. */
  headroomBytes: number
  /** Estimated decode throughput for the whole configuration. */
  decodeTokensPerSecond: number
  /** False only on the nearest miss reported when nothing fits. */
  fits: boolean
}

export interface InferenceResult {
  shape: ModelShape
  /** The format the whole checkpoint is named in, before the expert split. */
  weightFormat: WeightFormatId
  /** The split the answer was costed with, which is mixed when the buckets differ. */
  weightQuantization: WeightQuantization
  /** Bytes for every weight in the checkpoint. */
  weightsBytes: number
  /** Bytes held by the routed and shared experts. */
  expertWeightBytes: number
  /** Bytes held by attention, the dense feed forward, the router, and the embeddings. */
  denseWeightBytes: number
  /** Bytes one token reads on the forward pass, the figure the roofline divides. */
  activeBytesPerToken: number
  /** The dtype the KV cache was costed in. */
  kvCacheDtype: DtypeId
  /** The dtype a sparse indexer cache was costed in. */
  indexerDtype: DtypeId

  // --- What was asked for ------------------------------------------------
  contextLength: number
  sequences: number
  /** The card limit the ranking was given. */
  maxGpus: number

  // --- Memory -------------------------------------------------------------
  kvCacheBytes: number
  /** Bytes for each token in each sequence, summed over every layer. */
  kvBytesPerToken: number
  activationBytes: number
  runtimeReserveBytes: number
  totalBytes: number

  // --- The recommendation ------------------------------------------------
  verdict: InferenceVerdict
  /** The configuration the ranking chose, or null when nothing fits. */
  recommended: InferenceCandidate | null
  /** Every other configuration that fits, best first. */
  alternatives: InferenceCandidate[]
  /**
   * The nearest miss, reported only when nothing fits, so the page can name the
   * shortfall instead of showing an empty answer.
   */
  closest: InferenceCandidate | null

  // --- Throughput --------------------------------------------------------
  /** The head the rate was costed with, or none. */
  mtpHead: MtpHeadType
  /** The multiplier the selected head applies to the decode rate. */
  mtpSpeedup: number
  /** Estimated decode tokens each second for the whole configuration. */
  decodeTokensPerSecond: number
  /** The same figure for one sequence. */
  perSequenceTokensPerSecond: number
  /**
   * The decode rate with no head, which is the bandwidth roofline.
   *
   * Reported beside the adjusted rate so a reader can see what the head
   * added rather than only the total.
   */
  baseDecodeTokensPerSecond: number
  /** The roofline rate for one sequence. */
  basePerSequenceTokensPerSecond: number

  // --- Room to grow ------------------------------------------------------
  /** VRAM the recommendation leaves free on one card. */
  kvHeadroomBytes: number
  /** Context length the free VRAM would allow at the current sequence count. */
  maxContextAtSequences: number | null
  /** Sequence count the free VRAM would allow at the current context length. */
  maxSequencesAtContext: number | null

  steps: Array<{ label: string; detail: string }>
  constants: Array<{ key: string; value: number | string; source: string }>
  assumptions: string[]
  /** True when a value had to be inferred rather than read. */
  bestEffort: boolean
  notes: string[]
}

/** The estimator input a validation error belongs to, when it has one. */
export type InferenceInputField =
  | 'weightFormat'
  | 'mtpHead'
  | 'contextLength'
  | 'sequences'
  | 'headroom'
  | 'maxGpus'

export class InferenceInputError extends Error {
  /** The field to mark invalid in the form, or null when no single field owns it. */
  readonly field: InferenceInputField | null

  constructor(message: string, field: InferenceInputField | null = null) {
    super(message)
    this.name = 'InferenceInputError'
    this.field = field
  }
}
