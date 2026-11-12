import { describe, expect, it } from 'vitest'

import { analyzeTokenizer } from './tokenizer'

/**
 * Fixtures are hand-built tokenizer.json objects small enough to reason about
 * exactly. The reference `tokenizers` library validated the same algorithm on
 * the real GPT-2, Llama 2, Llama 3, Qwen, Mistral, Gemma, and DeepSeek
 * tokenizers; these synthetic files pin the behaviour in the repo so the
 * analysis cannot silently drift.
 */

/** A GPT-2-style byte-level BPE: ByteLevel pretokenizer and pair merges. */
const BYTE_LEVEL = {
  pre_tokenizer: { type: 'ByteLevel' },
  model: {
    vocab: {
      H: 0, e: 1, l: 2, o: 3, 'Ġ': 4, w: 5, r: 6, d: 7,
      Hello: 8, 'Ġworld': 9,
    },
    merges: ['H e', 'He l', 'Hel l', 'Hell o', 'Ġ w', 'Ġw o', 'Ġwo r', 'Ġwor l', 'Ġworl d'],
  },
}

/** A SentencePiece-style BPE with the ▁ marker and a byte fallback. */
const SENTENCEPIECE = {
  normalizer: {
    type: 'Sequence',
    normalizers: [
      { type: 'Prepend', prepend: '▁' },
      { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
    ],
  },
  model: {
    type: 'BPE',
    vocab: { '▁': 0, '▁a': 1, b: 2, '▁b': 3 },
    merges: ['▁ a', '▁ b'],
  },
}

/** The same SentencePiece tokenizer with byte fallback turned on. */
const SENTENCEPIECE_BYTE_FALLBACK = {
  ...SENTENCEPIECE,
  model: { ...SENTENCEPIECE.model, byte_fallback: true },
}

/** A unigram tokenizer in the id-keyed shape the hub stores. */
const UNIGRAM = {
  added_tokens: [{ content: '<pad>', special: true }],
  model: {
    type: 'Unigram',
    vocab: {
      0: ['<pad>', 0],
      1: ['▁', -0.5],
      2: ['a', -0.7],
      3: ['▁the', -1.2],
    },
  },
}

const WORDPIECE = {
  model: {
    type: 'WordPiece',
    vocab: { '[CLS]': 0, '[UNK]': 1, hello: 2, '##ing': 3, '##ly': 4, good: 5 },
  },
}

describe('analyzeTokenizer on a byte-level BPE', () => {
  it('splits words with the GPT-2 regex and replays the merges', () => {
    const profile = analyzeTokenizer(BYTE_LEVEL, 'Hello world')

    expect(profile.family).toBe('byte_level_bpe')
    // "Hello" and "Ġworld" each merge down to a single symbol.
    expect(profile.corpusTokens).toBe(2)
    expect(profile.corpusBytes).toBe(11)
    expect(profile.bytesPerToken).toBeCloseTo(5.5, 9)
    expect(profile.estimated).toBe(false)
    expect(profile.vocabSize).toBe(10)
  })

  it('counts a symbol not in the vocabulary as one token when there is no byte fallback', () => {
    const profile = analyzeTokenizer(BYTE_LEVEL, 'xyz')

    // Each letter survives as its own symbol, none of which is in the vocab.
    expect(profile.corpusTokens).toBe(3)
    expect(profile.corpusBytes).toBe(3)
    expect(profile.bytesPerToken).toBeCloseTo(1, 9)
  })
})

describe('analyzeTokenizer on a SentencePiece-style BPE', () => {
  it('normalises spaces to ▁ and splits by the merges', () => {
    const profile = analyzeTokenizer(SENTENCEPIECE, 'a b中')

    // "a b中" → "▁a▁b中" → symbols "▁a", "▁b", "中".
    expect(profile.family).toBe('sentencepiece_bpe')
    expect(profile.corpusTokens).toBe(3)
    expect(profile.corpusBytes).toBe(6)
    expect(profile.bytesPerToken).toBeCloseTo(2, 9)
  })

  it('expands an out-of-vocabulary character to one token per byte', () => {
    const profile = analyzeTokenizer(SENTENCEPIECE_BYTE_FALLBACK, 'a b中')

    // 中 is outside the vocab, so byte fallback writes it as three <0xXX> tokens.
    expect(profile.corpusTokens).toBe(5)
    expect(profile.bytesPerToken).toBeCloseTo(6 / 5, 9)
  })
})

describe('analyzeTokenizer on a unigram tokenizer', () => {
  it('weights piece lengths by the softmaxed scores', () => {
    const profile = analyzeTokenizer(UNIGRAM)

    expect(profile.family).toBe('unigram')
    expect(profile.corpusTokens).toBeNull()
    expect(profile.estimated).toBe(false)

    // Scores -0.5, -0.7, -1.2 over pieces of byte length 1, 1, and 4.
    const w1 = Math.exp(-0.5 + 0.5)
    const wa = Math.exp(-0.7 + 0.5)
    const wth = Math.exp(-1.2 + 0.5)
    const expected = (w1 * 1 + wa * 1 + wth * 4) / (w1 + wa + wth)
    expect(profile.bytesPerToken).toBeCloseTo(expected, 9)
  })

  it('ignores the special tokens when weighing', () => {
    // <pad> carries the top score, so if it were counted the bytes-per-token
    // figure would be pulled up by its five bytes.
    const profile = analyzeTokenizer(UNIGRAM)
    expect(profile.bytesPerToken).toBeLessThan(4)
  })
})

describe('analyzeTokenizer on a WordPiece tokenizer', () => {
  it('reports the mean decoded piece length as an estimate', () => {
    const profile = analyzeTokenizer(WORDPIECE)

    expect(profile.family).toBe('wordpiece')
    expect(profile.estimated).toBe(true)
    expect(profile.corpusTokens).toBeNull()
    // [CLS]=5 [UNK]=5 hello=5 ing=3 ly=2 good=4, averaged over six entries.
    expect(profile.bytesPerToken).toBeCloseTo(4, 9)
  })
})

describe('analyzeTokenizer detection and failures', () => {
  it('reads the vocabulary size including special tokens', () => {
    expect(analyzeTokenizer(BYTE_LEVEL, 'x').vocabSize).toBe(10)
    expect(analyzeTokenizer(WORDPIECE).vocabSize).toBe(6)
  })

  it('rejects a file with no vocabulary', () => {
    expect(() => analyzeTokenizer({})).toThrow(/No vocabulary/)
  })

  it('falls back to an estimate when nothing identifies the tokenizer', () => {
    const profile = analyzeTokenizer({ model: { vocab: { the: 0, cat: 1 }, merges: ['t h'] } })
    expect(profile.family).toBe('unknown')
    expect(profile.estimated).toBe(true)
  })
})