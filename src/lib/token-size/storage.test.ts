import { describe, expect, it } from 'vitest'

import {
  averageTokenIdDigits,
  bytesPerTokenForStorage,
  getStorageFormat,
  STORAGE_FORMAT_ORDER,
  STORAGE_FORMATS,
  storageNote,
  tokenIdWidth,
} from './storage'
import type { TokenizerProfile } from './types'

/** A minimal profile; the storage maths only reads the vocabulary and the text ratio. */
function profile(patch: Partial<TokenizerProfile> = {}): TokenizerProfile {
  return {
    family: 'byte_level_bpe',
    vocabSize: 32_000,
    corpusTokens: 1_000,
    corpusBytes: 4_000,
    bytesPerToken: 4,
    estimated: false,
    ...patch,
  }
}

describe('tokenIdWidth', () => {
  it('holds a vocabulary that fits in 16 bits at two bytes a token', () => {
    expect(tokenIdWidth(1)).toBe(2)
    expect(tokenIdWidth(50_257)).toBe(2)
    expect(tokenIdWidth(65_536)).toBe(2)
  })

  it('widens to four bytes once the vocabulary passes 16 bits', () => {
    expect(tokenIdWidth(65_537)).toBe(4)
    expect(tokenIdWidth(151_936)).toBe(4)
  })
})

describe('averageTokenIdDigits', () => {
  it('counts one digit for a single digit vocabulary', () => {
    expect(averageTokenIdDigits(5)).toBe(1)
    expect(averageTokenIdDigits(10)).toBe(1)
  })

  it('weighs each id length by how many ids have it', () => {
    // Ids 0-9 take one digit and 10-99 take two, so 100 ids average 1.9.
    expect(averageTokenIdDigits(100)).toBeCloseTo(1.9, 9)
    // Ids 0-9 take one digit, 10-99 two, and 100-149 three, so 150 ids
    // average (10 + 180 + 150) / 150.
    expect(averageTokenIdDigits(150)).toBeCloseTo(340 / 150, 9)
  })
})

describe('bytesPerTokenForStorage', () => {
  it('keeps the tokenizer measurement for text', () => {
    expect(bytesPerTokenForStorage('text', profile({ bytesPerToken: 4.447 }))).toBeCloseTo(4.447, 9)
  })

  it('prices Arrow at the raw id width', () => {
    expect(bytesPerTokenForStorage('arrow', profile({ vocabSize: 32_000 }))).toBe(2)
    expect(bytesPerTokenForStorage('arrow', profile({ vocabSize: 151_936 }))).toBe(4)
  })

  it('prices Parquet at the compressed share of the id width', () => {
    expect(bytesPerTokenForStorage('parquet', profile({ vocabSize: 32_000 }))).toBeCloseTo(1.2, 9)
    expect(bytesPerTokenForStorage('parquet', profile({ vocabSize: 151_936 }))).toBeCloseTo(2.4, 9)
  })

  it('prices JSONL at the mean id width plus its separator', () => {
    expect(bytesPerTokenForStorage('jsonl', profile({ vocabSize: 5 }))).toBe(2)
    expect(bytesPerTokenForStorage('jsonl', profile({ vocabSize: 100 }))).toBeCloseTo(2.9, 9)
  })
})

describe('storageNote', () => {
  it('explains each format in its own terms', () => {
    expect(storageNote('text', profile())).toContain('4.000 bytes')
    expect(storageNote('arrow', profile({ vocabSize: 151_936 }))).toContain('32-bit')
    expect(storageNote('parquet', profile({ vocabSize: 32_000 }))).toContain('1.20 bytes per token')
    expect(storageNote('jsonl', profile({ vocabSize: 100 }))).toContain('1.9 digits')
  })
})

describe('STORAGE_FORMATS', () => {
  it('lists every format in the declared order with a label and a hint', () => {
    expect(STORAGE_FORMAT_ORDER).toEqual(['text', 'jsonl', 'parquet', 'arrow'])
    expect(STORAGE_FORMATS.map((format) => format.id)).toEqual(STORAGE_FORMAT_ORDER)
    for (const format of STORAGE_FORMATS) {
      expect(format.label).not.toBe('')
      expect(format.hint).not.toBe('')
    }
  })

  it('looks a format up by id', () => {
    expect(getStorageFormat('parquet').label).toBe('Parquet')
  })
})
