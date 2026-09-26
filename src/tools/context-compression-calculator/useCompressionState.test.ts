import { describe, expect, it } from 'vitest'

import { defaultsFromSchema, serializeState } from '@/lib/url-state'

import { COMPRESSION_SCHEMA, type CompressionFormInputs } from './useCompressionState'

const DEFAULTS = defaultsFromSchema(COMPRESSION_SCHEMA)

describe('COMPRESSION_SCHEMA', () => {
  it('covers every field the page reads', () => {
    const expected: Array<keyof CompressionFormInputs> = [
      'inputPrice',
      'cachedInputPrice',
      'outputPrice',
      'missPercent',
      'cachePercent',
      'outputPercent',
      'compressionPercent',
    ]
    expect(Object.keys(COMPRESSION_SCHEMA).sort()).toEqual([...expected].sort())
  })

  it('opens on the Sonnet tier, the coding mix, and a 30 percent summary', () => {
    expect(DEFAULTS.inputPrice).toBe('3')
    expect(DEFAULTS.cachedInputPrice).toBe('0.3')
    expect(DEFAULTS.outputPrice).toBe('15')
    expect(DEFAULTS.missPercent).toBe('4.5')
    expect(DEFAULTS.cachePercent).toBe('95')
    expect(DEFAULTS.outputPercent).toBe('0.5')
    expect(DEFAULTS.compressionPercent).toBe('30')
  })

  it('writes nothing into the query string at the defaults', () => {
    expect(serializeState(COMPRESSION_SCHEMA, DEFAULTS)).toBe('')
  })

  it('round-trips every field through the query string', () => {
    const values: CompressionFormInputs = {
      inputPrice: '7',
      cachedInputPrice: '0.7',
      outputPrice: '21',
      missPercent: '12',
      cachePercent: '84',
      outputPercent: '4',
      compressionPercent: '8',
    }

    const params = new URLSearchParams(serializeState(COMPRESSION_SCHEMA, values))

    for (const key of Object.keys(COMPRESSION_SCHEMA) as Array<keyof CompressionFormInputs>) {
      const spec = COMPRESSION_SCHEMA[key]
      const raw = params.get(spec.param)
      expect(raw, `${String(key)} was not written`).not.toBeNull()
      expect(spec.parse(raw), `${String(key)} did not parse back`).toEqual(values[key])
    }
  })

  it('keeps a field at its default when the URL supplies nothing usable', () => {
    // A parser returns null for a sign, a fraction in the wrong place, or text,
    // and the hook then leaves the field alone rather than accepting it.
    expect(COMPRESSION_SCHEMA.inputPrice.parse('-1')).toBeNull()
    expect(COMPRESSION_SCHEMA.inputPrice.parse('abc')).toBeNull()
    expect(COMPRESSION_SCHEMA.cachedInputPrice.parse('')).toBeNull()
    expect(COMPRESSION_SCHEMA.inputPrice.parse(null)).toBeNull()
  })

  it('accepts a decimal rate and a decimal share', () => {
    expect(COMPRESSION_SCHEMA.cachedInputPrice.parse('0.0028')).toBe('0.0028')
    expect(COMPRESSION_SCHEMA.missPercent.parse('4.5')).toBe('4.5')
    expect(COMPRESSION_SCHEMA.compressionPercent.parse('12.5')).toBe('12.5')
  })

  it('keeps every parameter name short and distinct', () => {
    const params = Object.values(COMPRESSION_SCHEMA).map((spec) => spec.param)
    expect(new Set(params).size).toBe(params.length)
    for (const param of params) {
      expect(param.length).toBeLessThanOrEqual(6)
    }
  })
})
