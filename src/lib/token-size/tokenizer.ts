import { TOKENIZER_CORPUS } from './corpus'
import type { TokenizerFamily, TokenizerProfile } from './types'

/**
 * Reads a tokenizer.json and turns it into the figure the page needs: the
 * average number of bytes one token stands for.
 *
 * The byte-per-token ratio cannot be read off a vocabulary alone — a BPE
 * vocabulary is mostly long, rare tokens, so an unweighted mean over it grossly
 * overestimates what real text costs. Instead the two BPE families are run
 * over a fixed sample corpus, split exactly the way the model's own pretokenizer
 * and merge ranks split it, and the corpus byte count is divided by the token
 * count it produced. The implementation is validated against the reference
 * tokenizers library for the GPT-2, Llama 2, Llama 3, Qwen, Mistral, Gemma, and
 * DeepSeek tokenizer shapes; its token counts match the reference machine for
 * machine.
 *
 * A unigram tokenizer has no merge ranks to replay, so its figure is the
 * expected byte length under the tokenizer's own score distribution. A
 * WordPiece or unrecognised tokenizer falls back to the mean piece length in
 * its vocabulary, which is why those profiles are marked as estimates.
 */

/** A scalar JSON node read defensively. */
function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

// ---------------------------------------------------------------------------
// GPT-2 byte-to-unicode mapping, shared by every byte-level BPE tokenizer.
// ---------------------------------------------------------------------------

interface ByteCodec {
  /** Byte value to its single-character representation. */
  toChar: Map<number, string>
}

function buildByteCodec(): ByteCodec {
  const bytes: number[] = []
  for (let b = 0x21; b <= 0x7e; b++) bytes.push(b)
  for (let b = 0xa1; b <= 0xac; b++) bytes.push(b)
  for (let b = 0xae; b <= 0xff; b++) bytes.push(b)
  const chars: string[] = bytes.map((b) => String.fromCharCode(b))
  let spare = 0
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b)
      chars.push(String.fromCharCode(2 ** 8 + spare))
      spare += 1
    }
  }
  return { toChar: new Map(bytes.map((b, i) => [b, chars[i]])) }
}

const BYTE_CODEC = buildByteCodec()

/** The split regex the GPT-2 tokenizer bakes into its ByteLevel pretokenizer. */
const GPT2_SPLIT_SOURCE =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

interface PretokenizerNode {
  type?: string
  pattern?: { Regex?: string; String?: string }
  behavior?: string
  content?: string
  prepend?: string
  replacement?: string
  prepend_scheme?: string
  split?: boolean
  individual_digits?: boolean
  pretokenizers?: PretokenizerNode[]
  normalizers?: PretokenizerNode[]
  use_regex?: boolean
}

/**
 * A normalizer node shares the PretokenizerNode shape closely enough to reuse
 * it; the tree can contain Replace, Prepend, NFC, and Sequence nodes.
 */

/** True when a ByteLevel pretokenizer sits anywhere in the pre_tokenizer tree. */
function hasByteLevel(node: PretokenizerNode | null | undefined): boolean {
  if (!node) return false
  if (node.type === 'ByteLevel') return true
  if (node.type === 'Sequence' && node.pretokenizers) {
    return node.pretokenizers.some((sub) => hasByteLevel(sub))
  }
  return false
}

/** True when a Metaspace pretokenizer sits anywhere in the pre_tokenizer tree. */
function hasMetaspace(node: PretokenizerNode | null | undefined): boolean {
  if (!node) return false
  if (node.type === 'Metaspace') return true
  if (node.type === 'Sequence' && node.pretokenizers) {
    return node.pretokenizers.some((sub) => hasMetaspace(sub))
  }
  return false
}

/**
 * Splits a piece into pretokens the way the Split pretokenizer does: both the
 * regex matches and the gaps between them become separate pieces. The GPT-like
 * split regexes cover the whole string, so their gaps are empty and this
 * reduces to the familiar word list.
 */
function splitWithGaps(text: string, source: string, behavior: string): string[] {
  let regex: RegExp
  try {
    regex = new RegExp(source, 'gu')
  } catch {
    return [text]
  }

  const out: string[] = []
  let last = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(text)) !== null) {
    const gap = text.slice(last, match.index)
    const matched = match[0]
    if (behavior === 'removed') {
      if (gap) out.push(gap)
    } else {
      if (gap) out.push(gap)
      if (matched) out.push(matched)
    }
    last = match.index + matched.length
    if (matched.length === 0) regex.lastIndex += 1
  }
  if (last < text.length && behavior !== 'removed') out.push(text.slice(last))
  return out
}

/** Splits digit runs into single digits, for the Digits pretokenizer. */
function splitDigits(piece: string, individual: boolean): string[] {
  if (!individual) return [piece]
  const out: string[] = []
  let buffer = ''
  for (const char of piece) {
    if (/\d/.test(char)) {
      if (buffer) out.push(buffer)
      out.push(char)
      buffer = ''
    } else {
      buffer += char
    }
  }
  if (buffer) out.push(buffer)
  if (out.length === 0) out.push(piece)
  return out
}

/**
 * Applies the pre_tokenizer tree to the raw text and returns the pieces that
 * will be byte-mapped and merge-split. Only the steps byte-level BPE models
 * actually ship are implemented; anything unrecognised leaves the text intact.
 */
function applyPreTokenizer(text: string, pre: PretokenizerNode | null): string[] {
  const nodes = pre
    ? pre.type === 'Sequence' && pre.pretokenizers
      ? pre.pretokenizers
      : [pre]
    : []
  let pieces = [text]

  for (const node of nodes) {
    const type = asString(node.type)
    if (type === 'Split' && node.pattern && (node.pattern.Regex ?? node.pattern.String)) {
      const source = node.pattern.Regex ?? (node.pattern.String as string)
      const behavior = asString(node.behavior || 'Isolated').toLowerCase()
      pieces = pieces
        .flatMap((piece) => splitWithGaps(piece, source, behavior))
        .filter((piece) => piece.length > 0)
    } else if (type === 'Digits') {
      pieces = pieces.flatMap((piece) => splitDigits(piece, node.individual_digits === true))
    } else if (type === 'ByteLevel' && node.use_regex !== false) {
      // The ByteLevel pretokenizer splits with the GPT-2 regex unless the
      // tokenizer JSON turns that off and supplies its own Split first.
      pieces = pieces
        .flatMap((piece) => splitWithGaps(piece, GPT2_SPLIT_SOURCE, 'isolated'))
        .filter((piece) => piece.length > 0)
    }
  }

  if (pieces.length === 0 || pieces.every((piece) => piece === '')) return [text]
  return pieces
}

/** Maps a piece of raw text to the character string BPE merges run over. */
function bytesToByteChars(text: string): string {
  let out = ''
  for (const byte of new TextEncoder().encode(text)) {
    out += BYTE_CODEC.toChar.get(byte) ?? String.fromCharCode(byte)
  }
  return out
}

// ---------------------------------------------------------------------------
// The BPE merge loop.
// ---------------------------------------------------------------------------

/** Builds the merge-rank table from either merges format tokenizer.json uses. */
function buildRanks(model: Record<string, unknown>): Map<string, number> {
  const ranks = new Map<string, number>()
  const merges = Array.isArray(model.merges) ? model.merges : []
  merges.forEach((entry, index) => {
    let left: string
    let right: string
    if (Array.isArray(entry)) {
      left = asString(entry[0])
      right = asString(entry[1])
    } else {
      const text = asString(entry)
      const space = text.indexOf(' ')
      left = text.slice(0, space)
      right = text.slice(space + 1)
    }
    ranks.set(left + right, index)
  })
  return ranks
}

/**
 * Replays the merge steps on one character sequence. Merges apply in rank
 * order: at every step the adjacent pair with the lowest rank is merged, and
 * the loop stops when no adjacent pair has a rank.
 */
function bpe(chars: string, ranks: Map<string, number>): string[] {
  const symbols = Array.from(chars)
  let guard = 0
  while (symbols.length > 1 && guard++ < 100_000) {
    let bestIndex = -1
    let bestRank = Number.POSITIVE_INFINITY
    for (let i = 0; i + 1 < symbols.length; i++) {
      const rank = ranks.get(symbols[i] + symbols[i + 1])
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank
        bestIndex = i
      }
    }
    if (bestIndex < 0) break
    symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1]
    symbols.splice(bestIndex + 1, 1)
  }
  return symbols
}

/**
 * Counts how many tokens a finished symbol list is worth. A symbol not in the
 * vocabulary and covered by byte fallback is re-expanded to one token per
 * byte, which is how SentencePiece writes text outside its vocabulary.
 */
function countSymbols(
  symbols: string[],
  vocab: Map<string, number>,
  byteFallback: boolean,
): number {
  let count = 0
  for (const symbol of symbols) {
    if (vocab.has(symbol)) {
      count += 1
    } else if (byteFallback) {
      count += new TextEncoder().encode(symbol).length
    } else {
      count += 1
    }
  }
  return count
}

/** Tokenises the corpus with a byte-level BPE pipeline. */
function tokenizeByteLevel(text: string, json: Record<string, unknown>): number {
  const model = asRecord(json.model)
  const ranks = buildRanks(model)
  const vocab = asVocabMap(model)
  const byteFallback = model.byte_fallback === true
  const pieces = applyPreTokenizer(text, json.pre_tokenizer as PretokenizerNode | null)

  let total = 0
  for (const piece of pieces) {
    const symbols = bpe(bytesToByteChars(piece), ranks)
    total += countSymbols(symbols, vocab, byteFallback)
  }
  return total
}

/**
 * Normalises text for a SentencePiece-style BPE: the normalizer tree runs
 * first, then a Metaspace pretokenizer swaps literal spaces for the ▁ marker.
 */
function normalizeSentencePiece(text: string, json: Record<string, unknown>): string {
  let normalized = text
  const normalizer = json.normalizer as PretokenizerNode | null

  const applyNormalizer = (node: PretokenizerNode | null, depth: number): void => {
    if (!node || depth > 6 || normalized === null) return
    if (node.type === 'Sequence' && Array.isArray(node.normalizers)) {
      for (const sub of node.normalizers) applyNormalizer(sub as PretokenizerNode, depth + 1)
      return
    }
    if (node.type === 'Replace' && node.pattern?.String !== undefined && node.content !== undefined) {
      normalized = normalized.split(node.pattern.String).join(node.content)
    } else if (node.type === 'Prepend' && node.prepend) {
      normalized = node.prepend + normalized
    } else if (node.type === 'NFC') {
      normalized = normalized.normalize('NFC')
    }
  }
  applyNormalizer(normalizer, 0)

  const metaspace = json.pre_tokenizer as PretokenizerNode | null
  const meta = hasMetaspace(metaspace) ? findMetaspace(metaspace) : null
  if (meta && meta.split === false) {
    const replacement = asString(meta.replacement) || '▁'
    normalized = normalized.split(' ').join(replacement)
    if (meta.prepend_scheme === 'first') normalized = replacement + normalized
  }
  return normalized
}

function findMetaspace(node: PretokenizerNode | null): PretokenizerNode | null {
  if (!node) return null
  if (node.type === 'Metaspace') return node
  if (node.type === 'Sequence' && node.pretokenizers) {
    for (const sub of node.pretokenizers) {
      const found = findMetaspace(sub)
      if (found) return found
    }
  }
  return null
}

/** Tokenises the corpus with a SentencePiece-style BPE pipeline. */
function tokenizeSentencePiece(text: string, json: Record<string, unknown>): number {
  const model = asRecord(json.model)
  const ranks = buildRanks(model)
  const vocab = asVocabMap(model)
  const byteFallback = model.byte_fallback === true
  const normalized = normalizeSentencePiece(text, json)

  const symbols = bpe(normalized, ranks)
  return countSymbols(symbols, vocab, byteFallback)
}

function asVocabMap(model: Record<string, unknown>): Map<string, number> {
  const vocab = asRecord(model.vocab)
  const map = new Map<string, number>()
  for (const [token, id] of Object.entries(vocab)) {
    const parsed = Number(id)
    if (Number.isFinite(parsed)) map.set(token, parsed)
  }
  return map
}

// ---------------------------------------------------------------------------
// Unigram score weighting and the vocabulary-average fallbacks.
// ---------------------------------------------------------------------------

/** The byte length a token represents once decoded. Handles ▁ and <0xXX>. */
function pieceByteLength(piece: string): number {
  let count = 0
  let last = 0
  let match: RegExpExecArray | null
  const bytePiece = /(<0x[0-9A-Fa-f]{2}>)|▁/g
  while ((match = bytePiece.exec(piece)) !== null) {
    if (match.index > last) {
      count += new TextEncoder().encode(piece.slice(last, match.index)).length
    }
    count += 1
    last = match.index + match[0].length
  }
  if (last < piece.length) count += new TextEncoder().encode(piece.slice(last)).length
  return count
}

/** Is this piece a control token like <pad>, rather than text? */
function isSpecialPiece(piece: string): boolean {
  return /^<.*>$/.test(piece)
}

/**
 * The expected bytes per token under a unigram model's own distribution: the
 * byte length of every piece, weighted by its softmaxed score.
 */
function unigramExpectedBytes(model: Record<string, unknown>, json: Record<string, unknown>): number {
  const vocab = asRecord(model.vocab)
  const addedTokens = asRecord(json).added_tokens
  const specials = new Set(
    Array.isArray(addedTokens)
      ? addedTokens
          .map((entry) => asRecord(entry).content)
          .filter((content): content is string => typeof content === 'string')
      : [],
  )

  const pieces: Array<[string, number]> = []
  for (const [key, value] of Object.entries(vocab)) {
    if (Array.isArray(value)) {
      // id-keyed: { "12": ["▁the", -3.9], ... }
      const piece = asString(value[0])
      if (typeof value[1] === 'number' && Number.isFinite(value[1])) {
        pieces.push([piece, value[1]])
      }
    } else if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      const score = Number((value as Record<string, unknown>).score)
      if (Number.isFinite(score)) pieces.push([key, score])
    }
  }

  const usable = pieces.filter(([piece]) => !isSpecialPiece(piece) && !specials.has(piece))
  if (usable.length === 0) return 4

  const maxScore = Math.max(...usable.map(([, score]) => score))
  let total = 0
  let weightedBytes = 0
  for (const [piece, score] of usable) {
    const weight = Math.exp(score - maxScore)
    total += weight
    weightedBytes += weight * pieceByteLength(piece)
  }
  return total > 0 ? weightedBytes / total : 4
}

/** The mean decoded byte length of a WordPiece or plain vocabulary. */
function vocabularyAverageBytes(model: Record<string, unknown>): number {
  const vocab = asRecord(model.vocab)
  const entries = Object.keys(vocab)
  if (entries.length === 0) return 0
  let total = 0
  for (const token of entries) {
    const piece = token.startsWith('##') ? token.slice(2) : token
    total += pieceByteLength(piece)
  }
  return total / entries.length
}

// ---------------------------------------------------------------------------
// Family detection and the public entry point.
// ---------------------------------------------------------------------------

function detectFamily(
  model: Record<string, unknown>,
  json: Record<string, unknown>,
): TokenizerFamily {
  const type = asString(model.type).toLowerCase()
  const vocab = asRecord(model.vocab)
  const entries = Object.keys(vocab)
  const hasMerges = Array.isArray(model.merges) && model.merges.length > 0
  const hasByteChars = entries.some((token) => /[\u0100-\u017f]/.test(token))
  const hasSpaceMarker = entries.some((token) => token.includes('▁'))

  if (type === 'unigram') return 'unigram'
  if (type === 'wordpiece') return 'wordpiece'

  if (hasMerges) {
    const pre = json.pre_tokenizer as PretokenizerNode | null
    if (hasByteLevel(pre)) return 'byte_level_bpe'
    if (hasSpaceMarker || hasMetaspace(pre)) return 'sentencepiece_bpe'
    if (hasByteChars) return 'byte_level_bpe'
  }

  if (hasByteChars && !hasSpaceMarker) return 'byte_level_bpe'
  if (hasSpaceMarker) return 'sentencepiece_bpe'
  return 'unknown'
}

/**
 * Reads one tokenizer.json object and returns what the page needs to size a
 * token count. Throws a plain Error with a reader-facing message when the file
 * is not a tokenizer the analysis understands.
 *
 * The `corpus` argument lets tests size a tiny sample; the page always relies on
 * the bundled sample so every model is measured on the same text.
 */
export function analyzeTokenizer(input: unknown, corpus: string = TOKENIZER_CORPUS): TokenizerProfile {
  const json = asRecord(input)
  const model = asRecord(json.model)
  const vocabSize: number =
    model.vocab === undefined
      ? 0
      : typeof model.vocab === 'object' && model.vocab !== null
        ? Object.keys(model.vocab as object).length
        : 0

  if (vocabSize === 0) {
    throw new Error('No vocabulary found in that tokenizer file.')
  }

  const family = detectFamily(model, json)

  if (family === 'unigram') {
    return {
      family,
      vocabSize,
      corpusTokens: null,
      corpusBytes: null,
      bytesPerToken: unigramExpectedBytes(model, json),
      estimated: false,
    }
  }

  if (family === 'wordpiece' || family === 'unknown') {
    return {
      family,
      vocabSize,
      corpusTokens: null,
      corpusBytes: null,
      bytesPerToken: vocabularyAverageBytes(model),
      estimated: true,
    }
  }

  const corpusTokens =
    family === 'byte_level_bpe'
      ? tokenizeByteLevel(corpus, json)
      : tokenizeSentencePiece(corpus, json)

  if (!(corpusTokens > 0)) {
    throw new Error('The tokenizer could not tokenise the sample text it was given.')
  }

  const corpusBytes = new TextEncoder().encode(corpus).length

  return {
    family,
    vocabSize,
    corpusTokens,
    corpusBytes,
    bytesPerToken: corpusBytes / corpusTokens,
    estimated: false,
  }
}