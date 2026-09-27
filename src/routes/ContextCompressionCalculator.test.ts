import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'

import { COMPRESSION_FAQ, WORKLOAD_PRESETS, estimateCompression } from '@/lib/compression'
import { defaultsFromSchema } from '@/lib/url-state'

import ContextCompressionCalculator from './ContextCompressionCalculator'
import CompressionChart from '@/tools/context-compression-calculator/CompressionChart'
import CompressionForm from '@/tools/context-compression-calculator/CompressionForm'
import CompressionResults from '@/tools/context-compression-calculator/CompressionResults'
import {
  COMPRESSION_SCHEMA,
  type CompressionFormInputs,
} from '@/tools/context-compression-calculator/useCompressionState'

const PATH = '/tools/context-compression-calculator/'

/**
 * The page as a server would render it.
 *
 * Effects do not run here, so this covers the first paint, which is the page a
 * reader lands on and the prerendered HTML a crawler reads.
 */
function renderPage(search = ''): string {
  return renderToString(
    createElement(
      MemoryRouter,
      { initialEntries: [`${PATH}${search}`] },
      createElement(ContextCompressionCalculator),
    ),
  )
}

const html = renderPage()

describe('ContextCompressionCalculator at the first paint', () => {
  it('opens with the trail back to the tool index and its own heading', () => {
    expect(html).toContain('Context Compression Calculator')
    expect(html).toContain('href="/"')
  })

  it('shows the three groups of inputs', () => {
    expect(html).toContain('What you pay')
    expect(html).toContain('How your sessions spend tokens')
    expect(html).toContain('How far you compress')
    for (const id of [
      'compression-input-price',
      'compression-cached-price',
      'compression-output-price',
      'compression-miss',
      'compression-cache',
      'compression-output-share',
      'compression-percent',
      'compression-percent-range',
    ]) {
      expect(html).toContain(`id="${id}"`)
    }
  })

  it('offers every workload preset and the selected price list', () => {
    for (const preset of WORKLOAD_PRESETS) {
      expect(html).toContain(preset.label)
    }
    expect(html).toContain('Price list')
    // The closed select shows only the entry it holds. The rest of the list is
    // mounted when the popup opens, which a server render never does.
    expect(html).toContain('Anthropic Sonnet tier')
    expect(html).toContain('Custom')
  })

  it('reads the default prices and the default mix', () => {
    // The Anthropic Sonnet tier and the Normal coding mix, at a 10 percent cap.
    expect(html).toContain('value="4.5"')
    expect(html).toContain('value="95"')
    expect(html).toContain('value="0.5"')
    expect(html).toContain('value="10"')
  })

  it('ignores the query string, because it is read after mount', () => {
    // This is what keeps the prerendered markup and the first client render in
    // agreement. The read itself is covered in useCompressionState.test.ts.
    const shared = renderPage('?in=7&cin=0.7&out=21&miss=12&cache=84&outtok=4&comp=8')
    expect(shared).toContain('value="3"')
    expect(shared).toContain('value="10"')
  })

  it('answers with the session size where compressing starts to pay', () => {
    // The kept rate is 4.5 percent of the context at 3 dollars, 95 percent at
    // 0.30, and the reply at 15, over the 99.5 percent of the mix that is input.
    // A 10 percent cap is 100k summary tokens at 3.0754 dollars for 1M, which is
    // 0.3075 dollars, and 0.3075 over the kept rate of 0.49749 per token is
    // 618k session tokens.
    expect(html).toContain('Compress once the session passes')
    expect(html).toContain('618k tokens')
    expect(html).toContain('Compress once the session passes 618k tokens')
    expect(html).toContain('16.2 percent of the window')
    expect(html).toContain('$0.497')
    expect(html).toContain('$0.308')
  })

  it('says so plainly when no session size reaches the crossing point', () => {
    // A 30 percent cap on the coding mix breaks even at 1.85M tokens, which is
    // past the window, so there is no size at which it pays.
    const wide = renderToString(
      createElement(CompressionResults, {
        result: estimateCompression({
          inputPrice: 3,
          cachedInputPrice: 0.3,
          outputPrice: 15,
          mix: { missPercent: 4.5, cachePercent: 95, outputPercent: 0.5 },
          compressionPercent: 30,
        }),
        computeError: null,
      }),
    )

    expect(wide).toContain('never, at this cap')
    expect(wide).toContain('Compressing does not pay at a 30 percent cap')
    expect(wide).not.toContain('1.9M tokens')
  })

  it('draws both charts with labelled axes and a spoken description', () => {
    expect(html).toContain('Session size, in tokens of context')
    expect(html).toContain('Summary cap, as a share of the window')
    expect(html).toContain('role="img"')
    expect(html).toContain('aria-label=')
    // The dashed drop line marks the crossing point on the session chart.
    expect(html).toContain('stroke-dasharray')
  })

  it('never prints a break-even share above a full window', () => {
    // A cached price higher than the fresh price makes a summary cheaper than a
    // session token, so the crossing sits past 100 percent and no cap on the
    // page can be quoted as the widest one that pays.
    const inverted = renderToString(
      createElement(CompressionResults, {
        result: estimateCompression({
          inputPrice: 3,
          cachedInputPrice: 5,
          outputPrice: 15,
          mix: { missPercent: 10, cachePercent: 90, outputPercent: 1 },
          compressionPercent: 10,
        }),
        computeError: null,
      }),
    )

    expect(inverted).toContain('Every summary cap up to a full window pays off here')
    expect(inverted).not.toContain('157')
  })

  it('does not claim a crossing when the flat line stays above the climbing line', () => {
    // At a 30 percent cap on the coding mix the crossing is past the window, so
    // the session chart has no point to mark and no crossing to describe.
    const wide = renderToString(
      createElement(CompressionChart, {
        result: estimateCompression({
          inputPrice: 3,
          cachedInputPrice: 0.3,
          outputPrice: 15,
          mix: { missPercent: 4.5, cachePercent: 95, outputPercent: 0.5 },
          compressionPercent: 30,
        }),
      }),
    )

    expect(wide).toContain('The flat line stays above the climbing line for the whole window')
    expect(wide).not.toContain('Where they cross is the size')
  })

  it('shows the working behind the break-even share', () => {
    expect(html).toContain('How this was calculated')
    expect(html).toContain('The two rates in detail')
    expect(html).toContain('Values this used')
    expect(html).toContain('Assumptions and caveats')
  })

  it('carries the questions the page answers', () => {
    expect(html).toContain('Questions about compressing a session')
    for (const item of COMPRESSION_FAQ) {
      expect(html).toContain(item.question)
    }
  })
})

describe('the preset pickers', () => {
  const defaults = defaultsFromSchema(COMPRESSION_SCHEMA)

  function renderForm(patch: Partial<CompressionFormInputs>): string {
    return renderToString(
      createElement(CompressionForm, {
        inputs: { ...defaults, ...patch },
        update: () => {},
        invalidField: null,
        computeError: null,
      }),
    )
  }

  it('names the price list the three rates describe', () => {
    expect(renderForm({})).toContain('Anthropic Sonnet tier')
    expect(
      renderForm({ inputPrice: '0.14', cachedInputPrice: '0.0028', outputPrice: '0.28' }),
    ).toContain('DeepSeek Flash')
  })

  it('falls back to the custom entry once a rate is edited', () => {
    // Editing one rate of a preset is the case the reader hits first, and the
    // select has to stop claiming the preset it no longer matches.
    expect(renderForm({ inputPrice: '3.5' })).toContain('Your own rates')
    expect(renderForm({ inputPrice: '3.5' })).not.toContain('Anthropic Sonnet tier')
  })

  it('marks the workload preset the three shares describe', () => {
    const bugHunting = renderForm({ missPercent: '15', cachePercent: '80', outputPercent: '5' })
    expect(bugHunting).toContain('aria-pressed="true"')
    expect(bugHunting).toContain('You paste new logs and new failures each turn')

    const custom = renderForm({ missPercent: '15' })
    expect(custom).toContain('Your own token mix.')
  })
})

describe('ContextCompressionCalculator when the estimator rejects an input', () => {
  it('replaces the figures with the message rather than a stale number', () => {
    // The query string is applied after mount, so the page is driven through the
    // form props directly to reach the rejected state a reader would see.
    const rejected = renderToString(
      createElement(CompressionResults, {
        result: null,
        computeError: 'The input price must be more than zero.',
      }),
    )

    expect(rejected).toContain('The input price must be more than zero.')
    expect(rejected).not.toContain('Compress once the session passes')
  })
})

describe('CompressionForm when the estimator rejects an input', () => {
  const defaults = defaultsFromSchema(COMPRESSION_SCHEMA)

  it('marks the field and shows the message', () => {
    const rejected = renderToString(
      createElement(CompressionForm, {
        inputs: { ...defaults, compressionPercent: '150' },
        update: () => {},
        invalidField: 'compressionPercent',
        computeError: 'The compression share must be between 1 and 100.',
      }),
    )

    expect(rejected).toContain('The compression share must be between 1 and 100.')
    expect(rejected).toContain('aria-invalid="true"')
    expect(rejected).toContain('role="alert"')
  })

  it('shows no alert while every input is usable', () => {
    const accepted = renderToString(
      createElement(CompressionForm, {
        inputs: defaults,
        update: () => {},
        invalidField: null,
        computeError: null,
      }),
    )

    expect(accepted).not.toContain('role="alert"')
    expect(accepted).not.toContain('aria-invalid="true"')
  })
})
