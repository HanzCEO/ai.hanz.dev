import { describe, expect, it } from 'vitest'

import { detectModelShape } from '@/lib/model-shape'
import { loadConfigFixture } from '@/test/fixtures'

import { detectWeightQuantization, weightBytesFor } from './detect'

describe('detectWeightQuantization', () => {
  it('reads a mixed DeepSeek V4 checkpoint as MXFP4 experts beside FP8 dense', () => {
    const quant = detectWeightQuantization(loadConfigFixture('deepseek-v4-pro'))
    expect(quant.experts).toBe('MXFP4')
    expect(quant.dense).toBe('FP8_E4M3')
    expect(quant.mixed).toBe(true)
    expect(quant.source).toBe('quantization_config')
    expect(quant.fp8BlockSize).toBe(128 * 128)
  })

  it('reads the same split from a nested DeepSeek V4.1 config', () => {
    const quant = detectWeightQuantization(loadConfigFixture('deepseek-v41-flash'))
    expect(quant.experts).toBe('MXFP4')
    expect(quant.dense).toBe('FP8_E4M3')
    expect(quant.mixed).toBe(true)
    expect(quant.fp8BlockSize).toBe(32 * 32)
  })

  it('reads a MiMo checkpoint as MXFP4 and notes the ignored layers', () => {
    for (const name of ['mimo-v26-pro-rl', 'mimo-v26-flash-rl']) {
      const quant = detectWeightQuantization(loadConfigFixture(name))
      expect(quant.experts, name).toBe('MXFP4')
      expect(quant.dense, name).toBe('MXFP4')
      expect(quant.mixed, name).toBe(false)
      expect(quant.source, name).toBe('store_dtype')
      expect(quant.note ?? '', name).toContain('ignored')
      expect(quant.fp8BlockSize, name).toBeNull()
    }
  })

  it('reads a checkpoint with no quantization as BF16', () => {
    const quant = detectWeightQuantization(loadConfigFixture('qwen3-8b'))
    expect(quant.primary).toBe('BF16')
    expect(quant.experts).toBe('BF16')
    expect(quant.dense).toBe('BF16')
    expect(quant.mixed).toBe(false)
    expect(quant.note ?? '').toContain('BF16')
  })

  it('reads an empty config as the BF16 default', () => {
    const quant = detectWeightQuantization({})
    expect(quant.primary).toBe('BF16')
    expect(quant.source).toBe('assumed')
    expect(quant.mixed).toBe(false)
  })
})

describe('weightBytesFor', () => {
  it('splits a mixed checkpoint into two byte buckets that sum to the total', () => {
    const config = loadConfigFixture('deepseek-v4-pro')
    const shape = detectModelShape(config)
    expect(shape).not.toBeNull()
    if (!shape) return

    const bytes = weightBytesFor(shape, detectWeightQuantization(config))
    expect(bytes.experts + bytes.dense).toBeCloseTo(bytes.total, 6)
    expect(bytes.expertBytes).toBeCloseTo(0.53125, 10)
    expect(bytes.expertBytes).not.toBeCloseTo(bytes.denseBytes, 6)
  })

  it('costs a mixed checkpoint far below the same checkpoint in BF16', () => {
    const config = loadConfigFixture('deepseek-v4-pro')
    const shape = detectModelShape(config)
    expect(shape).not.toBeNull()
    if (!shape) return

    const bytes = weightBytesFor(shape, detectWeightQuantization(config))
    expect(bytes.total).toBeLessThan(shape.totalParams)
    expect(bytes.total).toBeLessThan(shape.totalParams * 2)
  })

  it('keeps the active byte count positive and below the whole checkpoint', () => {
    const config = loadConfigFixture('deepseek-v4-pro')
    const shape = detectModelShape(config)
    expect(shape).not.toBeNull()
    if (!shape) return

    const bytes = weightBytesFor(shape, detectWeightQuantization(config))
    expect(bytes.activePerToken).toBeGreaterThan(0)
    expect(bytes.activePerToken).toBeLessThan(bytes.total)
  })

  it('costs a dense BF16 checkpoint at two bytes for each weight', () => {
    const config = loadConfigFixture('qwen3-8b')
    const shape = detectModelShape(config)
    expect(shape).not.toBeNull()
    if (!shape) return

    const bytes = weightBytesFor(shape, detectWeightQuantization(config))
    expect(bytes.total).toBeCloseTo(shape.totalParams * 2, 3)
    expect(bytes.experts).toBe(0)
    expect(bytes.activePerToken).toBeCloseTo(shape.activeParamsPerToken * 2, 3)
  })
})
