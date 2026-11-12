import { describe, expect, it } from 'vitest'

import { bytesInDecimalUnits, estimateTokenSize, formatTokenAmountShort, parseTokenAmount } from './estimate'

describe('parseTokenAmount', () => {
  it('reads a plain whole number', () => {
    expect(parseTokenAmount('128000')).toBe(128_000)
    expect(parseTokenAmount(' 42 ')).toBe(42)
    expect(parseTokenAmount('128,000')).toBe(128_000)
  })

  it('reads a suffix the way people type token counts', () => {
    expect(parseTokenAmount('128k')).toBe(128_000)
    expect(parseTokenAmount('1K')).toBe(1_000)
    expect(parseTokenAmount('1.5M')).toBe(1_500_000)
    expect(parseTokenAmount('2b')).toBe(2_000_000_000)
    expect(parseTokenAmount('1T')).toBe(1_000_000_000_000)
  })

  it('rejects anything that is not a positive amount', () => {
    expect(parseTokenAmount('')).toBeNull()
    expect(parseTokenAmount('abc')).toBeNull()
    expect(parseTokenAmount('0')).toBeNull()
    expect(parseTokenAmount('-5')).toBeNull()
    expect(parseTokenAmount('1.5')).toBeNull()
    expect(parseTokenAmount('5x')).toBeNull()
    expect(parseTokenAmount('99999999999999999999')).toBeNull()
  })
})

describe('bytesInDecimalUnits', () => {
  it('gives every unit a 1000-based reading', () => {
    const lines = bytesInDecimalUnits(4_447_000_000)
    expect(lines.map((line) => line.unit)).toEqual(['B', 'KB', 'MB', 'GB', 'TB', 'PB'])
    expect(lines[0].value).toBe(4_447_000_000)
    expect(lines[3].value).toBeCloseTo(4.447, 9)
    expect(lines[4].value).toBeCloseTo(0.004447, 12)
  })
})

describe('estimateTokenSize', () => {
  it('scales the byte count by the tokenizer ratio', () => {
    const result = estimateTokenSize(1_000_000_000, 4.447)

    expect(result.bytes).toBeCloseTo(4_447_000_000, 6)
    expect(result.headline).toBe('4.45 GB')
    expect(result.exactBytes).toBe('4,447,000,000')
  })

  it('reports the binary equivalent from the same byte count', () => {
    const result = estimateTokenSize(1_000_000_000, 4)
    expect(result.binary).toMatch(/GiB/)
    // 4,000,000,000 bytes is 3.73 GiB.
    expect(result.binary).toContain('3.73')
  })

  it('falls back to a smaller unit for a small token count', () => {
    const result = estimateTokenSize(128_000, 4)
    expect(result.headline).toBe('512 KB')
  })
})

describe('formatTokenAmountShort', () => {
  it('fits a token amount on a button of one or two characters plus a suffix', () => {
    expect(formatTokenAmountShort(1_000)).toBe('1K')
    expect(formatTokenAmountShort(128_000)).toBe('128K')
    expect(formatTokenAmountShort(1_500_000)).toBe('1.5M')
    expect(formatTokenAmountShort(1_000_000_000)).toBe('1B')
    expect(formatTokenAmountShort(1_000_000_000_000)).toBe('1T')
  })
})