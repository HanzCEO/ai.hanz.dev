import { describe, expect, it } from 'vitest'

import { estimateCompression, keptCostAt, summaryCostAt } from './compute'
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

describe('keptCostAt', () => {
  it('scales a rate for 1M tokens by the token count', () => {
    expect(keptCostAt(3, 1_000_000)).toBe(3)
    expect(keptCostAt(3, 500_000)).toBe(1.5)
    expect(keptCostAt(0.3, 0)).toBe(0)
  })
})

describe('estimateCompression on the coding preset', () => {
  it('costs both paths at the Sonnet rates', () => {
    const result = estimateCompression(sonnet(CODING, 5))

    // 4.5 percent of the context at 3 dollars, 95 percent at 0.30, and the
    // reply at 15, all over the 99.5 percent of the mix that is input.
    expect(result.keptRatePerMillion).toBeCloseTo(0.49749, 4)
    // A summary token is new text at the full input price, plus its reply.
    expect(result.summaryUnitPerMillion).toBeCloseTo(3.0754, 4)
    expect(result.summaryTokens).toBe(50_000)
    expect(result.summaryCost).toBeCloseTo(0.15377, 5)
    expect(result.keptAtWindow).toBeCloseTo(0.49749, 4)
  })

  it('crosses at a session size just past 300k tokens', () => {
    const result = estimateCompression(sonnet(CODING, 5))

    expect(result.breakEvenSessionTokens).toBeCloseTo(309_091, 0)
    expect(result.compressingWins).toBe(true)
    expect(result.savingAtWindow).toBeGreaterThan(0)
  })

  it('splits the kept rate into the three parts that make it up', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const { miss, cache, output } = result.keptParts

    expect(miss + cache + output).toBeCloseTo(result.keptRatePerMillion, 12)
    expect(miss).toBeCloseTo((4.5 * 3) / 99.5, 12)
    expect(cache).toBeCloseTo((95 * 0.3) / 99.5, 12)
    expect(output).toBeCloseTo((0.5 * 15) / 99.5, 12)
  })

  it('splits the summary unit rate into input and output', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const { input, output } = result.summaryParts

    expect(input + output).toBeCloseTo(result.summaryUnitPerMillion, 12)
    expect(input).toBeCloseTo(3, 12)
    expect(result.outputPerInput).toBeCloseTo(0.5 / 99.5, 12)
  })

  it('normalizes a mix that does not add up to 100', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const doubled = estimateCompression(
      sonnet({ missPercent: 9, cachePercent: 190, outputPercent: 1 }, 5),
    )

    expect(result.normalized.missPercent).toBeCloseTo(4.5, 9)
    expect(doubled.keptRatePerMillion).toBeCloseTo(result.keptRatePerMillion, 12)
    expect(doubled.breakEvenSessionTokens).toBeCloseTo(result.breakEvenSessionTokens, 6)
  })
})

describe('the summary is capped at a share of the window', () => {
  it('does not change size with the session', () => {
    const result = estimateCompression(sonnet(CODING, 5))

    // Nothing in the inputs names a session, and the summary size is fixed by
    // the window alone, so the same result serves a short and a long session.
    expect(result.summaryTokens).toBe(
      (result.compressionPercent / 100) * result.contextWindowTokens,
    )
    expect(result.summaryTokens).toBe(50_000)
  })

  it('costs the same at any session size', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const atShort = summaryCostAt(result, 100_000)
    const atLong = summaryCostAt(result, 1_000_000)
    const atNothing = summaryCostAt(result)

    expect(atShort).toBe(atLong)
    expect(atNothing).toBe(atLong)
    expect(atLong).toBeCloseTo(result.summaryCost, 12)
    expect(result.summaryAtWindow).toBeCloseTo(result.summaryCost, 12)
  })

  it('scales the summary size with the cap', () => {
    const narrow = estimateCompression(sonnet(CODING, 2))
    const wide = estimateCompression(sonnet(CODING, 8))

    expect(narrow.summaryTokens).toBe(20_000)
    expect(wide.summaryTokens).toBe(80_000)
    expect(wide.summaryCost / narrow.summaryCost).toBeCloseTo(4, 9)
  })
})

describe('the crossing point', () => {
  it('matches the closed form window times cap over break-even', () => {
    const cases: Array<[WorkloadMix, number, number, number, number]> = [
      [CODING, 3, 0.3, 15, 5],
      [CODING, 2, 0.2, 12, 12],
      [BUG_HUNTING, 3, 0.3, 15, 5],
      [BUG_HUNTING, 0.14, 0.0028, 0.28, 9],
      [ASSISTANT, 1.4, 0.26, 4.4, 20],
    ]

    for (const [mix, inputPrice, cachedInputPrice, outputPrice, percent] of cases) {
      const result = estimateCompression({
        inputPrice,
        cachedInputPrice,
        outputPrice,
        mix,
        compressionPercent: percent,
      })
      const closedForm =
        (CONTEXT_WINDOW_TOKENS * percent) / result.breakEvenPercent

      expect(
        Math.abs(result.breakEvenSessionTokens - closedForm) /
          result.breakEvenSessionTokens,
      ).toBeLessThan(1e-9)
    }
  })

  it('puts the two costs level at the crossing point', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const kept = keptCostAt(result.keptRatePerMillion, result.breakEvenSessionTokens)

    expect(kept).toBeCloseTo(result.summaryCost, 9)
  })

  it('moves with the mix, because the kept rate moves', () => {
    const coding = estimateCompression(sonnet(CODING, 5))
    const hunting = estimateCompression(sonnet(BUG_HUNTING, 5))

    expect(hunting.breakEvenSessionTokens).toBeCloseTo(125_000, 0)
    expect(hunting.breakEvenSessionTokens).toBeLessThan(coding.breakEvenSessionTokens)
  })

  it('moves with the cap, later for a wider summary', () => {
    const narrow = estimateCompression(sonnet(CODING, 2))
    const wide = estimateCompression(sonnet(CODING, 8))

    expect(wide.breakEvenSessionTokens).toBeGreaterThan(narrow.breakEvenSessionTokens)
    expect(wide.breakEvenSessionTokens / narrow.breakEvenSessionTokens).toBeCloseTo(4, 9)
  })
})

describe('estimateCompression at the break-even share', () => {
  it('lands the crossing point on the full window', () => {
    const first = estimateCompression(sonnet(CODING, 5))
    const atBreakEven = estimateCompression(sonnet(CODING, first.breakEvenPercent))

    expect(atBreakEven.breakEvenSessionTokens).toBeCloseTo(CONTEXT_WINDOW_TOKENS, 0)
    expect(atBreakEven.keptAtWindow).toBeCloseTo(atBreakEven.summaryAtWindow, 9)
    expect(atBreakEven.savingAtWindow).toBeCloseTo(0, 9)
    // The crossing point is the window itself, so it is not inside it.
    expect(atBreakEven.compressingWins).toBe(false)
  })

  it('lands on the window for every preset and price list', () => {
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

        expect(
          Math.abs(
            atBreakEven.breakEvenSessionTokens - CONTEXT_WINDOW_TOKENS,
          ) / CONTEXT_WINDOW_TOKENS,
        ).toBeLessThan(1e-9)
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
    expect(below.breakEvenSessionTokens).toBeLessThan(CONTEXT_WINDOW_TOKENS)
    expect(above.compressingWins).toBe(false)
    expect(above.breakEvenSessionTokens).toBeGreaterThan(CONTEXT_WINDOW_TOKENS)
  })

  it('reports a crossing point past the window at a cap of 20', () => {
    const result = estimateCompression(sonnet(CODING, 20))

    expect(result.breakEvenPercent).toBeCloseTo(16.1765, 4)
    expect(result.compressingWins).toBe(false)
    expect(result.breakEvenSessionTokens).toBeGreaterThan(CONTEXT_WINDOW_TOKENS)
    expect(result.savingAtWindow).toBeLessThan(0)
  })

  it('never wins when the summary keeps the whole window', () => {
    // A cap of 100 is not summarising at all, and it still loses, because the
    // kept path pays the cheap cache price on most of its context and the
    // summary path pays the full input price on all of it.
    for (const workload of WORKLOAD_PRESETS) {
      const result = estimateCompression(sonnet(workload.mix, 100))
      expect(result.compressingWins).toBe(false)
    }
  })

  it('crosses inside the window for every preset at a cap of 1', () => {
    for (const workload of WORKLOAD_PRESETS) {
      const result = estimateCompression(sonnet(workload.mix, 1))
      expect(result.compressingWins).toBe(true)
      expect(result.breakEvenSessionTokens).toBeLessThan(CONTEXT_WINDOW_TOKENS)
    }
  })
})

describe('estimateCompression with a free cache', () => {
  it('pushes the crossing point later, because the kept path gets cheaper', () => {
    // The cheaper the cache, the more the kept path is worth and the harder a
    // summary has to work to beat it. So a free cache moves the crossing point
    // out rather than in.
    const paid = estimateCompression(sonnet(CODING, 5))
    const free = estimateCompression({ ...sonnet(CODING, 5), cachedInputPrice: 0 })

    expect(free.breakEvenSessionTokens).toBeGreaterThan(paid.breakEvenSessionTokens)
    expect(free.keptRatePerMillion).toBeLessThan(paid.keptRatePerMillion)
    // With a free cache the kept path pays only for what misses and for the
    // reply, so the crossing point is the summary cost over those two.
    expect(free.keptRatePerMillion).toBeCloseTo((4.5 * 3 + 0.5 * 15) / 99.5, 12)
  })

  it('pulls the crossing point in when the cache is dearer', () => {
    const cheap = estimateCompression(sonnet(CODING, 5))
    const dear = estimateCompression({ ...sonnet(CODING, 5), cachedInputPrice: 1 })

    expect(dear.breakEvenSessionTokens).toBeLessThan(cheap.breakEvenSessionTokens)
    expect(dear.keptRatePerMillion).toBeGreaterThan(cheap.keptRatePerMillion)
  })
})

describe('estimateCompression on the three workload presets', () => {
  it('accepts every preset', () => {
    for (const workload of WORKLOAD_PRESETS) {
      const result = estimateCompression(sonnet(workload.mix, 30))
      expect(Number.isFinite(result.keptRatePerMillion)).toBe(true)
      expect(Number.isFinite(result.breakEvenPercent)).toBe(true)
      expect(Number.isFinite(result.breakEvenSessionTokens)).toBe(true)
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

describe('the steps shown in the breakdown', () => {
  it('names the crossing as the point where the summary starts to pay', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const step = result.steps.find((entry) => entry.label === 'Session size where the two agree')

    expect(result.compressingWins).toBe(true)
    expect(step?.detail).toContain('it pays once the session passes')
    expect(step?.detail).toContain(
      Math.round(result.breakEvenSessionTokens).toLocaleString('en-US'),
    )
  })

  it('says the crossing is past the window when the cap never pays', () => {
    const result = estimateCompression(sonnet(CODING, 30))
    const step = result.steps.find((entry) => entry.label === 'Session size where the two agree')

    expect(result.compressingWins).toBe(false)
    expect(step?.detail).toContain('past the')
    expect(step?.detail).toContain('token window')
    expect(step?.detail).not.toContain('it pays once the session passes')
  })

  it('sizes the summary from the window the chart is drawn over', () => {
    const result = estimateCompression(sonnet(CODING, 5))
    const step = result.steps.find((entry) => entry.label === 'Size of the summary')

    expect(step?.detail).toContain('1,000,000 token window')
    expect(step?.detail).toContain('50,000 tokens')
    expect(result.summaryTokens).toBe(50_000)
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
  })
})
