import type { Provider } from '@/lib/kvcache'

/**
 * How a tokenizer turns text into tokens.
 *
 * The family decides how the byte-per-token figure is derived, and it is also
 * what the page reports beside the model id so a reader can see why two models
 * with the same vocabulary size still split text differently.
 */
export type TokenizerFamily = 'byte_level_bpe' | 'sentencepiece_bpe' | 'unigram' | 'wordpiece' | 'unknown'

export const TOKENIZER_FAMILY_LABELS: Record<TokenizerFamily, string> = {
  byte_level_bpe: 'byte-level BPE',
  sentencepiece_bpe: 'SentencePiece BPE',
  unigram: 'unigram',
  wordpiece: 'WordPiece',
  unknown: 'unrecognised tokenizer',
}

/**
 * What the analysis could learn about one tokenizer.
 *
 * `bytesPerToken` is the number the page multiplies a token count by. For the
 * two BPE families it is measured by tokenising the bundled corpus, so it is
 * exact for that text. A unigram tokenizer has no merges to replay, so its
 * figure is the expected byte length under its own score distribution. A
 * WordPiece or unrecognised tokenizer has no better source than the mean piece
 * length in its vocabulary, which is why `estimated` marks those two.
 */
export interface TokenizerProfile {
  family: TokenizerFamily
  /** Entries in the model vocabulary, including special tokens. */
  vocabSize: number
  /** Tokens the corpus split into, or null when the family was not tokenised. */
  corpusTokens: number | null
  /** Bytes in the corpus, or null when the family was not tokenised. */
  corpusBytes: number | null
  bytesPerToken: number
  /** True when bytesPerToken is a vocabulary average rather than a measurement. */
  estimated: boolean
}

/** A size expressed in one decimal unit. */
export interface SizeLine {
  unit: 'B' | 'KB' | 'MB' | 'GB' | 'TB' | 'PB'
  value: number
  text: string
}

/** The answer: a token count priced in bytes, then scaled into every unit. */
export interface TokenSizeResult {
  tokens: number
  bytes: number
  bytesPerToken: number
  /** The size in the unit that fits the magnitude, for example "4.45 GB". */
  headline: string
  /** Every decimal unit, from bytes to petabytes. */
  lines: SizeLine[]
  /** The same size in binary units, for example "4.14 GiB". */
  binary: string
  /** The exact byte count with thousands separators. */
  exactBytes: string
}

export type TokenizerErrorKind =
  | 'not_found'
  | 'gated'
  | 'malformed'
  | 'network'
  | 'unsupported'
  | 'unknown'

/** A failure worth showing a reader, with the fields the page needs to react. */
export class TokenizerError extends Error {
  readonly kind: TokenizerErrorKind
  readonly provider: Provider
  readonly repo: string

  constructor(
    message: string,
    options: {
      kind: TokenizerErrorKind
      provider: Provider
      repo: string
      cause?: unknown
    },
  ) {
    super(message, { cause: options.cause })
    this.name = 'TokenizerError'
    this.kind = options.kind
    this.provider = options.provider
    this.repo = options.repo
  }
}