import { describe, expect, it } from 'vitest'

import {
  DEFAULT_WEIGHT_FORMAT,
  WEIGHT_FORMATS,
  bytesPerWeight,
  getWeightFormat,
  isFourBitFormat,
  isFp8Format,
  isWeightFormatId,
  weightFormatLabel,
} from './formats'

describe('bytesPerWeight', () => {
  it('counts the scale sidecar in the 4 bit formats', () => {
    // One E8M0 scale for each 32 values on top of the 4 bit payload.
    expect(bytesPerWeight('MXFP4')).toBeCloseTo(0.53125, 10)
    // One E4M3 scale for each 16 values on top of the 4 bit payload.
    expect(bytesPerWeight('NVFP4')).toBeCloseTo(0.5625, 10)
    // A 16 bit scale and zero point for each group of 128 values.
    expect(bytesPerWeight('INT4')).toBeCloseTo(0.515625, 10)
  })

  it('keeps the two byte formats at two bytes', () => {
    expect(bytesPerWeight('BF16')).toBe(2)
    expect(bytesPerWeight('FP16')).toBe(2)
  })

  it('adds the FP8 scale at the block size the caller names', () => {
    expect(bytesPerWeight('FP8_E4M3')).toBeCloseTo(1 + 1 / 16384, 12)
    expect(bytesPerWeight('FP8_E4M3', { blockSize: 1024 })).toBeCloseTo(1 + 1 / 1024, 12)
    expect(bytesPerWeight('FP8_E5M2', { blockSize: 1024 })).toBeCloseTo(1 + 1 / 1024, 12)
  })

  it('leaves a format with no scale alone when a block size is passed', () => {
    expect(bytesPerWeight('MXFP4', { blockSize: 1024 })).toBeCloseTo(0.53125, 10)
    expect(bytesPerWeight('BF16', { blockSize: 1024 })).toBe(2)
  })

  it('keeps INT8 at one byte', () => {
    expect(bytesPerWeight('INT8')).toBe(1)
  })
})

describe('format table', () => {
  it('opens on BF16', () => {
    expect(DEFAULT_WEIGHT_FORMAT).toBe('BF16')
    expect(isWeightFormatId(DEFAULT_WEIGHT_FORMAT)).toBe(true)
  })

  it('catalogues every id once and gives each a label and a note', () => {
    const ids = WEIGHT_FORMATS.map((format) => format.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const format of WEIGHT_FORMATS) {
      expect(format.label.trim(), format.id).not.toBe('')
      expect(format.note.trim(), format.id).not.toBe('')
      expect(format.bytes, format.id).toBeGreaterThan(0)
    }
  })

  it('resolves a label and rejects an unknown id', () => {
    expect(weightFormatLabel('MXFP4')).toBe('MXFP4')
    expect(getWeightFormat('BF16').bytes).toBe(2)
    expect(isWeightFormatId('FP4')).toBe(false)
    expect(isWeightFormatId('')).toBe(false)
  })

  it('separates the 4 bit and FP8 families', () => {
    expect(isFourBitFormat('MXFP4')).toBe(true)
    expect(isFourBitFormat('NVFP4')).toBe(true)
    expect(isFourBitFormat('INT4')).toBe(true)
    expect(isFourBitFormat('FP8_E4M3')).toBe(false)
    expect(isFp8Format('FP8_E4M3')).toBe(true)
    expect(isFp8Format('FP8_E5M2')).toBe(true)
    expect(isFp8Format('MXFP4')).toBe(false)
  })
})
