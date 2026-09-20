import { describe, expect, it } from 'vitest'

import { formatBytes, formatDuration } from './format'

describe('formatDuration', () => {
  it('reports a sub second duration as less than a second', () => {
    // A quarter of a second of weight streaming is real work. Rounding it to
    // "0 seconds" made a fast store read as though it did nothing.
    expect(formatDuration(0.27)).toBe('less than a second')
    expect(formatDuration(0.001)).toBe('less than a second')
  })

  it('agrees the unit with the value', () => {
    // The seconds branch rounded to a whole number and then always said
    // "seconds", so a duration of just over one second read as "1 seconds".
    expect(formatDuration(1.09)).toBe('1 second')
    expect(formatDuration(8.59)).toBe('9 seconds')
    expect(formatDuration(1.2)).toBe('1 second')
    expect(formatDuration(1.6)).toBe('2 seconds')
    expect(formatDuration(89)).toBe('89 seconds')
  })

  it('keeps reporting no time for a duration that is not positive', () => {
    expect(formatDuration(0)).toBe('no time')
    expect(formatDuration(-5)).toBe('no time')
    expect(formatDuration(Number.NaN)).toBe('no time')
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('no time')
  })

  it('leaves the larger units alone', () => {
    // The minutes threshold starts at 90 seconds, so the smallest value it can
    // produce is 1.5 minutes. None of these branches can reach a singular value.
    expect(formatDuration(600)).toBe('10 minutes')
    expect(formatDuration(5400)).toBe('1.5 hours')
    expect(formatDuration(7200)).toBe('2.0 hours')
    expect(formatDuration(400_000)).toBe('4.6 days')
  })
})

describe('formatBytes', () => {
  it('reports a byte count with binary units', () => {
    expect(formatBytes(1536).text).toBe('1.50 KiB')
    expect(formatBytes(1024 ** 3).text).toBe('1.00 GiB')
  })

  it('clamps a non positive count to zero', () => {
    // This is what turned a zeroed cache size into a printable "0 B" in the
    // results panel. The clamp stays, because callers rely on it, but the
    // panels now branch before they interpolate a value that can be zero.
    expect(formatBytes(0).text).toBe('0 B')
    expect(formatBytes(-1).text).toBe('0 B')
    expect(formatBytes(Number.NaN).text).toBe('0 B')
  })
})
