import type { GpuSpec, StorageSpec } from '../hardware'
import type { WeightFormatId } from '../weight-format'

/**
 * Weight format the calibration pass reads and the pruned model is saved in.
 *
 * The ids are shared with every other calculator, so a checkpoint that
 * publishes MXFP4 or NVFP4 is named the same way here as it is on the inference
 * page. The calibration reads the whole checkpoint, but the VRAM peak is one
 * expert block, so the expert bucket is the one that decides the fit.
 */
export type WeightDtype = WeightFormatId

/** The MoE shape a config describes, in parameter counts. */
export interface MoeShape {
  modelType: string
  hiddenSize: number
  /** Dense feed forward width, used by the layers that are not MoE. */
  intermediateSize: number
  /** Expert feed forward width. */
  moeIntermediateSize: number
  /** Routed experts per MoE layer. */
  routedExperts: number
  /** Experts a single token is routed to, the top-k of the router. */
  expertsPerToken: number
  /** Always-on shared experts per MoE layer. */
  sharedExperts: number
  sharedExpertIntermediate: number
  /** Transformer blocks in total. */
  numLayers: number
  /** Blocks that contain an expert bank. Only these cost expert compute. */
  moeLayers: number
  /** Blocks whose feed forward is a plain dense MLP. */
  denseLayers: number

  /** Parameters in one attention block. */
  attentionParamsPerLayer: number
  /** Parameters in one dense MLP block. */
  denseFfnParamsPerLayer: number
  /** Parameters in one router. */
  routerParamsPerLayer: number
  /** Parameters in one routed expert, three projections. */
  paramsPerExpert: number
  /** Parameters in one shared expert. */
  paramsPerSharedExpert: number

  /** Parameters touched by one token, across every layer. Router excluded. */
  activeParamsPerToken: number
  /** Parameters that exist in the file, across every layer. */
  totalParams: number
  routedExpertParams: number
  sharedExpertParams: number
  attentionParams: number
  denseFfnParams: number
  embedParams: number

  /** True when a field had to be inferred rather than read directly. */
  bestEffort: boolean
  notes: string[]
}

export interface ReapInputs {
  /** Calibration samples fed through the model. */
  calibrationSamples: number
  /** Tokens per sample. The product with samples is the real cost driver. */
  sequenceLength: number
  /** Fraction of routed experts removed, between 0 and 0.9. */
  pruneRatio: number
  gpu: GpuSpec
  storage: StorageSpec
  /** Weight format the calibration pass reads. */
  weightDtype: WeightDtype
  /** Model factory utilisation, between 0.01 and 1. */
  mfu: number
  /** Multiplier over the pure compute or stream time. */
  overheadFactor: number
  /** Fixed cost in seconds: load, tokenize, write the pruned model. */
  setupSeconds: number
  /** Samples held in flight at once. Drives the activation buffer. */
  microBatchSize: number
  /** Whether the router top-k is reduced in proportion to the pruning ratio. */
  scaleTopK: boolean
}

export type ReapVerdict =
  | 'not-moe'
  | 'fits'
  /** The selected format overflows the card, but a narrower format fits it. */
  | 'needs-narrower'
  /** No weight format in the fitting search holds one expert block. */
  | 'needs-offload'

/** Whether the run is limited by the GPU or by the weight stream. */
export type ReapBound = 'compute' | 'stream'

export interface ReapResult {
  /** False when the model has no expert bank, in which case every figure is zero. */
  moe: boolean
  verdict: ReapVerdict
  shape: MoeShape | null

  tokens: number
  flops: number

  computeSeconds: number
  streamSeconds: number
  setupSeconds: number
  overheadFactor: number
  /** The figure to show a user: the larger of the two, plus overhead and setup. */
  estimateSeconds: number
  bound: ReapBound

  weightBytes: number
  /** One MoE block resident, which is the peak of the layer-wise observer. */
  perMoELayerBytes: number
  activationBytes: number
  peakVramBytes: number
  vramBytes: number
  /**
   * The widest weight format whose block fits, or null when none does. The
   * search runs from the widest format to the narrowest, so this is the highest
   * precision the card can hold and the format the verdict names.
   */
  bestFittingDtype: WeightDtype | null
  /** One expert block in that format, or null when no format fits. */
  bestFittingBlockBytes: number | null
  /** The format the calibration was costed in, which is what the reader selected. */
  weightDtype: WeightDtype

  pruneRatio: number
  keptExperts: number
  removedExperts: number
  expertsPerTokenAfter: number
  totalParamsAfter: number
  expertParamsAfter: number
  reductionPercent: number
  activeParamsPerTokenAfter: number

  steps: Array<{ label: string; detail: string }>
  constants: Array<{ key: string; value: number | string; source: string }>
  assumptions: string[]
}

/** The estimator input a validation error belongs to, when it has one. */
export type ReapInputField =
  | 'samples'
  | 'sequenceLength'
  | 'pruneRatio'
  | 'mfu'
  | 'overheadFactor'
  | 'microBatchSize'

export class ReapInputError extends Error {
  /** The field to mark invalid in the form, or null when no single field owns it. */
  readonly field: ReapInputField | null

  constructor(message: string, field: ReapInputField | null = null) {
    super(message)
    this.name = 'ReapInputError'
    this.field = field
  }
}
