/**
 * The shape of the context compression cost model.
 *
 * The question the page answers is when a summary starts to pay. A session has
 * a context, and the next request bills that context one of two ways. Keep it
 * and pay the cached rate on most of it, which grows with the session.
 * Summarise it down to a share A of the model window and pay the full input
 * rate on a fixed number of tokens, which does not grow at all.
 *
 * The kept cost is a line through the origin and the summary cost is a flat
 * line, so the two cross at exactly one session size. That crossing point is
 * the answer, and the share at which the summary stops paying is the second
 * number the page reports.
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
  /** Share of the model context window the summary is capped at, as a percentage. */
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
  /** Dollars for 1M session tokens when the whole session is kept. */
  keptRatePerMillion: number
  /** Dollars for 1M summary tokens, which no cache holds. */
  summaryUnitPerMillion: number
  /** Tokens the summary is capped at, which does not move with the session. */
  summaryTokens: number
  /** Cost of one request that carries the summary, the same at any session size. */
  summaryCost: number

  /** The session size at which the two paths cost the same, in tokens. */
  breakEvenSessionTokens: number
  /** The share of the window a summary may be capped at and still pay off. */
  breakEvenPercent: number
  /** True when the crossing point falls inside the window. */
  compressingWins: boolean

  /** Cost of one request at the full context window, kept whole. */
  keptAtWindow: number
  /** Cost of one request carrying the summary, which is flat. */
  summaryAtWindow: number
  /** keptAtWindow minus summaryAtWindow. Positive means the summary pays. */
  savingAtWindow: number

  /** Output tokens for each input token, which is o divided by (m + c). */
  outputPerInput: number

  /** Where each dollar of the kept rate goes. */
  keptParts: { miss: number; cache: number; output: number }
  /** Where each dollar of the summary unit rate goes. */
  summaryParts: { input: number; output: number }

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
