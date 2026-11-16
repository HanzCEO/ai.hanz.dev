import type { StorageFormatId, StorageFormatSpec, TokenizerProfile } from './types'

/**
 * The file formats a token count can be stored in.
 *
 * A running model consumes token ids, not text. Once a corpus has been
 * tokenized, the ids are written to a file the same way any other integers are:
 * as fixed width columns for Parquet and Arrow, or as decimal text for JSONL.
 * Each format therefore has its own bytes per token, and none of them match the
 * UTF-8 text the tokenizer was measured on. The page offers the choice so a
 * reader can size the file they actually keep.
 */
export const STORAGE_FORMATS: StorageFormatSpec[] = [
  {
    id: 'text',
    label: 'Text',
    hint: 'UTF-8 text measured with the model tokenizer, which is the reading the page is built on.',
  },
  {
    id: 'jsonl',
    label: 'JSONL',
    hint: 'Token ids written as decimal numbers and separated by commas, one document per line.',
  },
  {
    id: 'parquet',
    label: 'Parquet',
    hint: 'Token ids in a compressed columnar file. Compression usually brings an integer column to about 60% of its raw width.',
  },
  {
    id: 'arrow',
    label: 'Arrow',
    hint: 'Token ids in an uncompressed Arrow IPC column, at the fixed width the vocabulary requires.',
  },
]

const STORAGE_FORMAT_BY_ID = new Map(STORAGE_FORMATS.map((format) => [format.id, format]))

/** Every format id, in the order the picker lists them. */
export const STORAGE_FORMAT_ORDER: StorageFormatId[] = STORAGE_FORMATS.map((format) => format.id)

export function getStorageFormat(id: StorageFormatId): StorageFormatSpec {
  const spec = STORAGE_FORMAT_BY_ID.get(id)
  if (!spec) throw new Error(`Unknown storage format: ${id}`)
  return spec
}

/** One token id is an unsigned integer just wide enough for the vocabulary. */
export function tokenIdWidth(vocabSize: number): 2 | 4 {
  return vocabSize <= 65_536 ? 2 : 4
}

/**
 * Snappy on an integer token column typically lands near 60% of the
 * uncompressed width. Published tokenized corpora bear this out: a 16-bit
 * vocabulary column runs about 1.2 bytes per token, and a 32-bit one about 2.4.
 * The ratio is a rule of thumb, so the page names Parquet as an estimate.
 */
const PARQUET_COMPRESSION_RATIO = 0.6

/**
 * The mean decimal width of the ids in a vocabulary, used to price JSONL.
 *
 * Ids run from 0 to vocabSize - 1, so the mean counts how many of them have one
 * digit, two digits, and so on. A vocabulary of 100, for example, spends one
 * digit on 10 ids and two on the other 90, for a mean of 1.9.
 */
export function averageTokenIdDigits(vocabSize: number): number {
  const upperBound = Math.max(1, Math.floor(vocabSize))
  let digits = 0
  let ids = 0
  for (let width = 1; width <= String(upperBound).length; width++) {
    const lower = width === 1 ? 0 : 10 ** (width - 1)
    const upper = Math.min(upperBound, 10 ** width)
    if (upper <= lower) continue
    digits += (upper - lower) * width
    ids += upper - lower
  }
  return ids === 0 ? 1 : digits / ids
}

/**
 * The bytes one token costs when the token count is written in a format.
 *
 * Text uses the tokenizer's own measurement. The other formats store ids, so
 * they follow the vocabulary width, not the language.
 */
export function bytesPerTokenForStorage(
  format: StorageFormatId,
  profile: TokenizerProfile,
): number {
  switch (format) {
    case 'text':
      return profile.bytesPerToken
    case 'jsonl':
      // One byte for the comma that separates the id from the next one.
      return averageTokenIdDigits(profile.vocabSize) + 1
    case 'parquet':
      return tokenIdWidth(profile.vocabSize) * PARQUET_COMPRESSION_RATIO
    case 'arrow':
      return tokenIdWidth(profile.vocabSize)
  }
}

/** The sentence that explains how the selected format's figure was derived. */
export function storageNote(format: StorageFormatId, profile: TokenizerProfile): string {
  if (format === 'text') {
    if (profile.corpusTokens !== null && profile.corpusBytes !== null) {
      return `The tokenizer split the bundled sample corpus of ${profile.corpusBytes.toLocaleString('en-US')} bytes into ${profile.corpusTokens.toLocaleString('en-US')} tokens, which puts one token at ${profile.bytesPerToken.toFixed(3)} bytes.`
    }
    if (profile.family === 'unigram') {
      return `A unigram tokenizer has no merges to replay, so the figure is the expected byte length of a token under the tokenizer\u2019s own score distribution.`
    }
    return `The tokenizer has no merge ranks, so the figure is the mean decoded byte length of the tokens in its vocabulary, which is an estimate.`
  }

  const width = tokenIdWidth(profile.vocabSize)
  if (format === 'arrow') {
    return `Each token becomes an unsigned ${width * 8}-bit id in an uncompressed Arrow column, so one token costs ${width} bytes plus the file\u2019s small footer.`
  }
  if (format === 'parquet') {
    return `Each token becomes an unsigned ${width * 8}-bit id. Compression usually brings a Parquet token column to about ${(width * PARQUET_COMPRESSION_RATIO).toFixed(2)} bytes per token; your encoder and compression settings can move it either way.`
  }

  const digits = averageTokenIdDigits(profile.vocabSize)
  return `Each token id is written as decimal text and separated by a comma. An average id in this vocabulary takes ${digits.toFixed(1)} digits, so one token costs about ${bytesPerTokenForStorage('jsonl', profile).toFixed(2)} bytes.`
}
