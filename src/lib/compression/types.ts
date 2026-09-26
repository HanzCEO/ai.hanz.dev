/**
 * The shape of the context compression cost model.
 *
 * The question the page answers is narrow. A session has a context, and the
 * next request bills that context one of two ways: keep it and pay the cache
 * rate on most of it, or summarise it down to a share A and pay the full input
 * rate on what is left. Both are linear in the context size, so the whole
 * decision reduces to one number, the share A at which the two are equal.
 */

/** How a session spends its billed tokens, as percentages that add to 100. */
export interface WorkloadMix {
  /** Input tokens the provider bills at the full input price. */
  missPercent: number
  /** Input tokens the provider bills at the cached input price. */
  cachePercent: number
  /** Output tokens the model writes. */
  outputPercent: number
}

export interface CompressionInputs {
  /** Dollars for 1M input tokens that miss the cache. */
  inputPrice: number
  /** Dollars for 1M input tokens that hit the cache. */
  cachedInputPrice: number
  /** Dollars for 1M output tokens. */
  outputPrice: number
  mix: WorkloadMix
  /** Share of the context the summary keeps, as a percentage. */
  compressionPercent: number
}

/** The input a validation error belongs to, when it has one. */
export type CompressionInputField =
  | 'inputPrice'
  | 'cachedInputPrice'
  | 'outputPrice'
  | 'missPercent'
  | 'cachePercent'
  | 'outputPercent'
  | 'compressionPercent'

/** One arithmetic step, so the page can show its work. */
export interface CompressionStep {
  label: string
  detail: string
}

/** One value the model read, and where it came from. */
export interface CompressionConstant {
  key: string
  value: number | string
  source: string
}

export interface CompressionResult {
  /** Dollars for 1M context tokens when the whole session is kept. */
  neverRatePerMillion: number
  /** Dollars for 1M context tokens when the session is summarised first. */
  compressRatePerMillion: number
  /** The share the summary must stay above to pay off. */
  breakEvenPercent: number
  /** True when the chosen share is below the break-even share. */
  compressingWins: boolean

  /** Cost of one request at the full context window, kept whole. */
  neverAtWindow: number
  /** Cost of one request at the full context window, summarised first. */
  compressAtWindow: number
  /** neverAtWindow minus compressAtWindow. Positive means the summary pays. */
  savingAtWindow: number

  /** Output tokens for each input token, which is o divided by (m + c). */
  outputPerInput: number

  /** Where each dollar of the kept-context rate goes. */
  neverParts: { miss: number; cache: number; output: number }
  /** Where each dollar of the summarised rate goes. */
  compressParts: { input: number; output: number }

  /** The shares the model used, after they were normalized to sum to 100. */
  normalized: WorkloadMix

  compressionPercent: number
  contextWindowTokens: number

  steps: CompressionStep[]
  constants: CompressionConstant[]
  assumptions: string[]
}

export class CompressionInputError extends Error {
  /** The field to mark invalid in the form, or null when no single field owns it. */
  readonly field: CompressionInputField | null

  constructor(message: string, field: CompressionInputField | null = null) {
    super(message)
    this.name = 'CompressionInputError'
    this.field = field
  }
}
