import type { GpuSpec } from '../hardware'
import type { RawConfig } from '../model-config'
import type { ModelShape } from '../model-shape'

/**
 * The two precisions this calculator answers for.
 *
 * FP16 and BF16 both take two bytes for each weight, so the memory footprint is
 * identical and only the numeric range differs. A model served in either one
 * needs the same hardware, which is why they share a single code path and a
 * single bytes-per-weight constant.
 */
export type InferencePrecision = 'FP16' | 'BF16'

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
  precision: InferencePrecision
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
  precision: InferencePrecision
  /** Bytes for each weight. Two for both precisions. */
  bytesPerWeight: number

  // --- What was asked for ------------------------------------------------
  contextLength: number
  sequences: number
  /** The card limit the ranking was given. */
  maxGpus: number

  // --- Memory -------------------------------------------------------------
  weightsBytes: number
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
  /** Estimated decode tokens each second for the whole configuration. */
  decodeTokensPerSecond: number
  /** The same figure for one sequence. */
  perSequenceTokensPerSecond: number

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
  | 'precision'
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
