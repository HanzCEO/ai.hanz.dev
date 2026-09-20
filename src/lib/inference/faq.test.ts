import { describe, expect, it } from 'vitest'

import { INFERENCE_FAQ } from './faq'

/**
 * The page copy is held to the house style, so this suite checks the questions
 * the page declares as structured data rather than only the ones on screen.
 * One list feeds both, so a question that passes here passes everywhere.
 */

/** Splits an answer into sentences, so each one can be measured on its own. */
function sentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence !== '')
}

function wordCount(sentence: string): number {
  return sentence.split(/\s+/).filter((word) => word !== '').length
}

describe('INFERENCE_FAQ', () => {
  it('keeps the question a reader arrives with first', () => {
    expect(INFERENCE_FAQ[0].question).toBe('How much GPU memory does FP16 or BF16 inference need?')
  })

  it('answers the cache dtype and the two step question', () => {
    const questions = INFERENCE_FAQ.map((item) => item.question)
    expect(questions).toContain('Does the KV cache dtype change the GPU answer?')
    expect(questions).toContain('Why does the page ask for the KV cache first?')
  })

  it('stays a short list', () => {
    expect(INFERENCE_FAQ.length).toBeGreaterThanOrEqual(6)
    expect(INFERENCE_FAQ.length).toBeLessThanOrEqual(8)
  })

  it('asks one question at a time', () => {
    for (const item of INFERENCE_FAQ) {
      expect(item.question.endsWith('?')).toBe(true)
      expect(item.question.trim()).not.toBe('')
    }
  })

  it('keeps every sentence to 20 words or fewer', () => {
    for (const item of INFERENCE_FAQ) {
      for (const sentence of sentences(item.answer)) {
        expect(
          wordCount(sentence),
          `${wordCount(sentence)} words: ${sentence}`,
        ).toBeLessThanOrEqual(20)
      }
    }
  })

  it('uses no em dash or en dash', () => {
    for (const item of INFERENCE_FAQ) {
      expect(item.question).not.toContain('\u2014')
      expect(item.question).not.toContain('\u2013')
      expect(item.answer).not.toContain('\u2014')
      expect(item.answer).not.toContain('\u2013')
    }
  })

  it('uses no contraction', () => {
    for (const item of INFERENCE_FAQ) {
      for (const contraction of ["doesn't", "isn't", "it's", "can't", "won't", "aren't"]) {
        expect(item.question).not.toContain(contraction)
        expect(item.answer).not.toContain(contraction)
      }
    }
  })

  it('names the two precisions it answers for', () => {
    const all = INFERENCE_FAQ.map((item) => `${item.question} ${item.answer}`).join(' ')
    expect(all).toContain('FP16')
    expect(all).toContain('BF16')
  })
})
