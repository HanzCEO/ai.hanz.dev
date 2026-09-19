export type Provider = 'huggingface' | 'modelscope'

export type DtypeId =
  | 'BF16'
  | 'FP16'
  | 'FP32'
  | 'FP8_E4M3'
  | 'FP8_E5M2'
  | 'INT8'
  | 'INT4'
  | 'FP4'

export type SupportLevel = 'supported' | 'untested' | 'unsupported'

export type ArchitectureFamily = 'gqa' | 'mla' | 'hybrid_linear' | 'dsv4' | 'dsv41' | 'unknown'

/** A model config as returned by the hub. Fields are read defensively. */
export type RawConfig = Record<string, unknown>

export interface DtypeSpec {
  id: DtypeId
  label: string
  /** Bytes per element. Fractional for sub-byte formats. */
  bytes: number
}

export interface DtypeSupport {
  dtype: DtypeId
  level: SupportLevel
  /** Why this level was assigned. */
  reason: string
}

export interface ComputeOptions {
  contextLength: number
  sequenceCount: number
  kvCacheDtype: DtypeId
  indexerDtype: DtypeId
}

export interface ComponentBreakdown {
  id: 'attention' | 'indexer' | 'state'
  label: string
  /** Human readable formula for this component. */
  formula: string
  /** Bytes per token per sequence, summed over all layers. Zero when constant. */
  bytesPerToken: number
  /** Bytes for one sequence. */
  bytesPerSequence: number
  totalBytes: number
  /** Constants from the config that feed this component. */
  inputs: Array<{ key: string; value: number | string }>
  note?: string
}

export interface ConstantUsed {
  key: string
  value: number | string
  /** Where the value came from, for auditing. */
  source: string
}

export interface LayerSplit {
  total: number
  fullAttention: number
  slidingAttention: number
  linearAttention: number
  compressed?: Array<{ ratio: number; layers: number }>
}

export interface ComputeResult {
  totalBytes: number
  bytesPerToken: number
  architecture: {
    family: ArchitectureFamily
    /** model_type from the config. */
    modelType: string
    label: string
  }
  layerSplit: LayerSplit
  components: ComponentBreakdown[]
  constants: ConstantUsed[]
  /** Ordered arithmetic chain, for the breakdown panel. */
  steps: Array<{ label: string; detail: string }>
  assumptions: string[]
  /** True when the architecture was inferred rather than recognised. */
  bestEffort: boolean
  /** Support tags for the attention cache dtype dropdown. */
  dtypeSupport: DtypeSupport[]
  /** Support tags for the indexer dtype dropdown. Null when the model has no indexer. */
  indexerDtypeSupport: DtypeSupport[] | null
  contextLength: number
  sequenceCount: number
  maxPositionEmbeddings: number | null
}

export class KvCacheInputError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'KvCacheInputError'
  }
}

export type ConfigErrorKind =
  | 'not_found'
  | 'gated'
  | 'malformed'
  | 'network'
  | 'unknown'

export class ModelConfigError extends Error {
  readonly kind: ConfigErrorKind
  readonly provider: Provider
  readonly repo: string
  /** Set when the other provider is likely to have the model. */
  readonly suggestOtherProvider: boolean

  constructor(
    message: string,
    options: {
      kind: ConfigErrorKind
      provider: Provider
      repo: string
      suggestOtherProvider?: boolean
      cause?: unknown
    },
  ) {
    super(message, { cause: options.cause })
    this.name = 'ModelConfigError'
    this.kind = options.kind
    this.provider = options.provider
    this.repo = options.repo
    this.suggestOtherProvider = options.suggestOtherProvider ?? false
  }
}
