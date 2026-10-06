import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'

import { REACHABILITY_FAQ, REACHABILITY_MODELS } from '@/lib/reachability'

import ZeroShotClassificationReachability from './ZeroShotClassificationReachability'

/**
 * The page as a server renders it.
 *
 * Effects do not run here, so this covers the first paint: the table a reader
 * lands on before touching a control, and the markup the prerender step writes
 * for a crawler. The table is rendered from the baked dataset, so no fetch is
 * involved and the assertions are stable between runs.
 */
const PATH = '/tools/zero-shot-classification-reachability-leaderboard/'

const html = renderToString(
  createElement(
    MemoryRouter,
    { initialEntries: [PATH] },
    createElement(ZeroShotClassificationReachability),
  ),
)

/** The control ids every reader has to be able to reach. */
const CONTROL_IDS = [
  'reachability-search',
  'reachability-gpu-class',
  'reachability-vendor',
  'reachability-context',
  'reachability-sequences',
  'reachability-weight-format',
  'reachability-kv-dtype',
  'reachability-max-gpus',
  'reachability-sort',
]

describe('ZeroShotClassificationReachability at the first paint', () => {
  it('names the page', () => {
    expect(html).toContain('Zero-Shot Classification Reachability Leaderboard')
  })

  it('renders the board as a table', () => {
    expect(html).toContain('<table')
    expect(html).toContain('The board')
  })

  it('renders one row for every model on the board', () => {
    // Every row names the engine it belongs to, so the count of those markers
    // is the count of rows. The page renders from the baked dataset, so this
    // checks the table rather than a live fetch.
    const rows = html.match(/data-engine="/g) ?? []
    expect(rows).toHaveLength(REACHABILITY_MODELS.length)
  })

  it('shows a known model from the dataset', () => {
    const first = REACHABILITY_MODELS[0]
    expect(first).toBeDefined()
    expect(html).toContain(first.name)
  })

  it('names the Decision Index in full, so it is not read as the rank', () => {
    expect(html).toContain('Decision Index')
    expect(html).toContain('Runs on')
    expect(html).toContain('Holds it alone')
    expect(html).toContain('Tokens/s')
    expect(html).toContain('Weights')
  })

  it('says how many models were sized rather than how many were ranked', () => {
    expect(html).toContain(`of ${REACHABILITY_MODELS.length} models sized for this workload`)
  })

  it('carries the short reason on a row it could not size', () => {
    // The unsized rows are the ones with a label under the model name, so the
    // first of them is the closed hosted API.
    expect(html).toContain('Closed hosted API')
    expect(html).toContain('Not sized')
    // Those rows carry a dash in all three number cells rather than a measured
    // zero. Counting cells rather than matching a substring is deliberate: a
    // sized row can legitimately read "0 of 40" when no single card holds it.
    const unsized = html.match(/Not sized/g) ?? []
    const dashes = html.match(/>-<\/td>/g) ?? []
    expect(unsized.length).toBeGreaterThan(0)
    expect(dashes).toHaveLength(unsized.length * 3)
  })

  it('collapses the questions while keeping every answer in the document', () => {
    expect(html).toContain('<details')
    expect(html).not.toContain('<details open')
    for (const item of REACHABILITY_FAQ) {
      expect(html).toContain(item.answer)
    }
  })

  it('explains what the decode column counts', () => {
    expect(html).toContain('total across the concurrent sequences')
  })

  it('carries every workload control', () => {
    for (const id of CONTROL_IDS) {
      expect(html).toContain(`id="${id}"`)
    }
  })

  it('labels every control', () => {
    for (const label of [
      'Search',
      'Card class',
      'Vendor',
      'Context length',
      'Concurrent sequences',
      'Weight format',
      'KV cache dtype',
      'Maximum cards for one model',
      'Order',
    ]) {
      expect(html).toContain(label)
    }
  })

  it('carries the questions the page answers', () => {
    expect(html).toContain('Questions about reachability')
    for (const item of REACHABILITY_FAQ) {
      expect(html).toContain(item.question)
    }
  })
  it('cites where the numbers come from', () => {
    expect(html).toContain('Where the numbers come from')
    expect(html).toContain('clef-evals Decision Model Leaderboard')
  })

  it('uses no em dash or en dash', () => {
    expect(html).not.toContain('\u2014')
    expect(html).not.toContain('\u2013')
  })

  it('uses no contraction', () => {
    for (const contraction of ["doesn't", "isn't", "it's", "can't", "won't"]) {
      expect(html).not.toContain(contraction)
    }
  })
})
