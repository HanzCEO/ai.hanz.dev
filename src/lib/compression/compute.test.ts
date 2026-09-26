import { describe, expect, it } from 'vitest'

import { estimateCompression, requestCost } from './compute'
import {
  CONTEXT_WINDOW_TOKENS,
  PRICE_PRESETS,
  WORKLOAD_PRESETS,
  findWorkloadPreset,
} from './presets'
import { CompressionInputError, type CompressionInputs, type WorkloadMix } from './types'

const CODING: WorkloadMix = { missPercent: 4.5, cachePercent: 95, outputPercent: 0.5 }
const BUG_HUNTING: WorkloadMix = { missPercent: 15, cachePercent: 80, outputPercent: 5 }
const ASSISTANT: WorkloadMix = { missPercent: 10, cachePercent: 89, outputPercent: 1 }

/** The Anthropic Sonnet tier, which is the default price preset. */
function sonnet(mix: WorkloadMix, compressionPercent: number): CompressionInputs {
  return {
    inputPrice: 3,
    cachedInputPrice: 0.3,
    outputPrice: 15,
    mix,
    compressionPercent,
  }
}

describe('requestCost', () => {
  it('scales a rate for 1M tokens by the token count', () => {
    expect(requestCost(3, 1_000_000)).toBe(3)
    expect(requestCost(3, 500_000)).toBe(1.5)
    expect(requestCost(0.3, 0)).toBe(0)
  })
})

describe('estimateCompression on the coding preset', () => {
  it('costs both paths at the Sonnet rates', () => {
    const result = estimateCompression(sonnet(CODING, 30))

    // 4.5 percent of the context at 3 dollars, 95 percent at 0.30, and the
    // reply at 15, all over the 99.5 percent of the mix that is input.
    expect(result.neverRatePerMillion).toBeCloseTo(0.4975, 4)
    expect(result.compressRatePerMillion).toBeCloseTo(0.9226, 4)
    expect(result.breakEvenPercent).toBeCloseTo(16.1765, 4)
    expect(result.compressingWins).toBe(false)
  })

  it('splits the kept rate into the three parts that make it up', () => {
    const result = estimateCompression(sonnet(CODING, 30))
    const { miss, cache, output } = result.neverParts

    expect(miss + cache + output).toBeCloseTo(result.neverRatePerMillion, 12)
    expect(miss).toBeCloseTo((4.5 * 3) / 99.5, 12)
    expect(cache).toBeCloseTo((95 * 0.3) / 99.5, 12)
    expect(output).toBeCloseTo((0.5 * 15) / 99.5, 12)
  })

  it('splits the summarised rate into input and output', () => {
    const result = estimateCompression(sonnet(CODING, 30))
    const { input, output } = result.compressParts

    expect(input + output).toBeCloseTo(result.compressRatePerMillion, 12)
    expect(input).toBeCloseTo(0.9, 12)
    expect(result.outputPerInput).toBeCloseTo(0.5 / 99.5, 12)
  })

  it('costs one request at the full window', () => {
    const result = estimateCompression(sonnet(CODING, 30))

    expect(result.contextWindowTokens).toBe(CONTEXT_WINDOW_TOKENS)
    expect(result.neverAtWindow).toBeCloseTo(0.4975, 4)
    expect(result.compressAtWindow).toBeCloseTo(0.9226, 4)
    expect(result.savingAtWindow).toBeCloseTo(-0.4251, 4)
  })

  it('normalizes a mix that does not add up to 100', () => {
    const result = estimateCompression(
      sonnet({ missPercent: 4.5, cachePercent: 95, outputPercent: 0.5 }, 30),
    )
    const doubled = estimateCompression(
      sonnet({ missPercent: 9, cachePercent: 190, outputPercent: 1 }, 30),
    )

    expect(result.normalized.missPercent).toBeCloseTo(4.5, 9)
    expect(doubled.neverRatePerMillion).toBeCloseTo(result.neverRatePerMillion, 12)
    expect(doubled.breakEvenPercent).toBeCloseTo(result.breakEvenPercent, 12)
  })
})

describe('estimateCompression at the break-even share', () => {
  it('makes the two rates agree', () => {
    const first = estimateCompression(sonnet(CODING, 30))
    const atBreakEven = estimateCompression(sonnet(CODING, first.breakEvenPercent))

    expect(atBreakEven.compressRatePerMillion).toBeCloseTo(
      atBreakEven.neverRatePerMillion,
      9,
    )
    expect(atBreakEven.savingAtWindow).toBeCloseTo(0, 9)
  })

  it('agrees at the break-even share for every preset and price list', () => {
    for (const workload of WORKLOAD_PRESETS) {
      for (const price of PRICE_PRESETS) {
        const base: CompressionInputs = {
          inputPrice: price.inputPrice,
          cachedInputPrice: price.cachedInputPrice,
          outputPrice: price.outputPrice,
          mix: workload.mix,
          compressionPercent: 30,
        }
        const first = estimateCompression(base)
        const atBreakEven = estimateCompression({
          ...base,
          compressionPercent: first.breakEvenPercent,
        })

        expect(atBreakEven.compressRatePerMillion).toBeCloseTo(
          atBreakEven.neverRatePerMillion,
          9,
        )
      }
    }
  })
})

describe('estimateCompression on either side of the break-even share', () => {
  it('wins below the break-even share and loses above it', () => {
    const base = estimateCompression(sonnet(BUG_HUNTING, 30))
    const below = estimateCompression(sonnet(BUG_HUNTING, base.breakEvenPercent - 1))
    const above = estimateCompression(sonnet(BUG_HUNTING, base.breakEvenPercent + 1))

    expect(below.compressingWins).toBe(true)
    expect(below.savingAtWindow).toBeGreaterThan(0)
    expect(above.compressingWins).toBe(false)
    expect(above.savingAtWindow).toBeLessThan(0)
  })

  it('never wins when the summary keeps the whole context', () => {
    // A share of 100 is not summarising at all, and it still loses, because the
    // kept path pays the cheap cache price on most of its context and the
    // summarised path pays the full input price on all of it.
    for (const workload of WORKLOAD_PRESETS) {
      const result = estimateCompression(sonnet(workload.mix, 100))
      expect(result.compressingWins).toBe(false)
    }
  })

  it('loses only at a share of 100 when the cache price matches the input price', () => {
    // With no cache discount the kept path is barely cheaper than a summary,
    // so the break-even share sits at its ceiling of 100 and any real summary
    // wins. A share of 100 is not summarising at all, and the two are equal.
    const result = estimateCompression({
      ...sonnet(CODING, 1),
      cachedInputPrice: 3,
    })
    expect(result.breakEvenPercent).toBeCloseTo(100, 6)
    expect(result.compressingWins).toBe(true)
    expect(estimateCompression({ ...sonnet(CODING, 100), cachedInputPrice: 3 }).compressingWins).toBe(
      false,
    )
  })

  it('wins at a share of 1 for the bug hunting preset', () => {
    const result = estimateCompression(sonnet(BUG_HUNTING, 1))
    expect(result.breakEvenPercent).toBeCloseTo(40, 6)
    expect(result.compressingWins).toBe(true)
  })
})

describe('estimateCompression with a free cache', () => {
  it('lowers the break-even share, because the kept path gets cheaper', () => {
    // The cheaper the cache, the more the kept path is worth and the harder a
    // summary has to work to beat it. So a free cache pushes the break-even
    // share down rather than up.
    const paid = estimateCompression(sonnet(CODING, 30))
    const free = estimateCompression({ ...sonnet(CODING, 30), cachedInputPrice: 0 })

    expect(free.breakEvenPercent).toBeLessThan(paid.breakEvenPercent)
    expect(free.neverRatePerMillion).toBeLessThan(paid.neverRatePerMillion)
    // With a free cache the kept path pays only for what misses and for the
    // reply, so the break-even share is those two over the input price.
    expect(free.breakEvenPercent).toBeCloseTo(
      (100 * (4.5 * 3 + 0.5 * 15)) / (99.5 * 3 + 0.5 * 15),
      9,
    )
  })

  it('raises the break-even share when the cache is dearer', () => {
    const cheap = estimateCompression(sonnet(CODING, 30))
    const dear = estimateCompression({ ...sonnet(CODING, 30), cachedInputPrice: 1 })

    expect(dear.breakEvenPercent).toBeGreaterThan(cheap.breakEvenPercent)
    expect(dear.neverRatePerMillion).toBeGreaterThan(cheap.neverRatePerMillion)
  })
})

describe('estimateCompression on the three workload presets', () => {
  it('accepts every preset', () => {
    for (const workload of WORKLOAD_PRESETS) {
      const result = estimateCompression(sonnet(workload.mix, 30))
      expect(Number.isFinite(result.neverRatePerMillion)).toBe(true)
      expect(Number.isFinite(result.breakEvenPercent)).toBe(true)
    }
  })

  it('carries the percentages the user gave', () => {
    expect(findWorkloadPreset('coding')?.mix).toEqual(CODING)
    expect(findWorkloadPreset('bug-hunting')?.mix).toEqual(BUG_HUNTING)
    expect(findWorkloadPreset('assistant')?.mix).toEqual(ASSISTANT)
    expect(WORKLOAD_PRESETS.map((preset) => preset.label)).toEqual([
      'Normal coding',
      'Bug hunting',
      'Personal assistant',
    ])
  })
})

describe('estimateCompression validation', () => {
  function fieldOf(inputs: CompressionInputs): string | null {
    try {
      estimateCompression(inputs)
    } catch (error) {
      expect(error).toBeInstanceOf(CompressionInputError)
      return (error as CompressionInputError).field
    }
    throw new Error('The estimator accepted inputs it should have rejected.')
  }

  it('rejects a non positive input price', () => {
    expect(fieldOf({ ...sonnet(CODING, 30), inputPrice: 0 })).toBe('inputPrice')
    expect(fieldOf({ ...sonnet(CODING, 30), inputPrice: -1 })).toBe('inputPrice')
  })

  it('rejects an unusable price', () => {
    expect(fieldOf({ ...sonnet(CODING, 30), inputPrice: Number.NaN })).toBe('inputPrice')
    expect(fieldOf({ ...sonnet(CODING, 30), cachedInputPrice: Number.NaN })).toBe(
      'cachedInputPrice',
    )
    expect(fieldOf({ ...sonnet(CODING, 30), outputPrice: Number.POSITIVE_INFINITY })).toBe(
      'outputPrice',
    )
  })

  it('rejects a negative price', () => {
    expect(fieldOf({ ...sonnet(CODING, 30), cachedInputPrice: -0.1 })).toBe('cachedInputPrice')
    expect(fieldOf({ ...sonnet(CODING, 30), outputPrice: -1 })).toBe('outputPrice')
  })

  it('rejects a negative share', () => {
    expect(fieldOf(sonnet({ ...CODING, missPercent: -1 }, 30))).toBe('missPercent')
    expect(fieldOf(sonnet({ ...CODING, cachePercent: -1 }, 30))).toBe('cachePercent')
    expect(fieldOf(sonnet({ ...CODING, outputPercent: -1 }, 30))).toBe('outputPercent')
  })

  it('rejects an unusable share', () => {
    expect(fieldOf(sonnet({ ...CODING, missPercent: Number.NaN }, 30))).toBe('missPercent')
  })

  it('rejects a mix whose input side is empty', () => {
    // With no input tokens the context has no size, so there is no rate to
    // compare. No single field owns this, so the error carries no field.
    const mix: WorkloadMix = { missPercent: 0, cachePercent: 0, outputPercent: 100 }
    expect(fieldOf(sonnet(mix, 30))).toBe(null)
  })

  it('rejects a compression share outside its range', () => {
    expect(fieldOf(sonnet(CODING, 0))).toBe('compressionPercent')
    expect(fieldOf(sonnet(CODING, 101))).toBe('compressionPercent')
    expect(fieldOf(sonnet(CODING, Number.NaN))).toBe('compressionPercent')
  })

  it('accepts the ends of the compression range', () => {
    expect(estimateCompression(sonnet(CODING, 1)).compressionPercent).toBe(1)
    expect(estimateCompression(sonnet(CODING, 100)).compressionPercent).toBe(100)
  })
})

describe('estimateCompression explanation', () => {
  it('describes every step and states its assumptions', () => {
    const result = estimateCompression(sonnet(CODING, 30))

    expect(result.steps.length).toBeGreaterThanOrEqual(4)
    for (const step of result.steps) {
      expect(step.label.length).toBeGreaterThan(0)
      expect(step.detail.length).toBeGreaterThan(0)
    }
    expect(result.assumptions.length).toBeGreaterThan(0)
  })
})
