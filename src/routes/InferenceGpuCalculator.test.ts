import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'

import { INFERENCE_FAQ } from '@/lib/inference'

import InferenceGpuCalculator from './InferenceGpuCalculator'
import KvCacheCalculator from './KvCacheCalculator'

/**
 * The composed pages, rendered as a server would render them.
 *
 * Effects do not run here, so this covers the first paint: the page a reader
 * lands on before any fetch resolves, and the prerendered HTML a crawler reads.
 */
function renderPage(page: React.ReactElement, path: string): string {
  return renderToString(createElement(MemoryRouter, { initialEntries: [path] }, page))
}

describe('InferenceGpuCalculator at the first paint', () => {
  const html = renderPage(createElement(InferenceGpuCalculator), '/tools/inference-gpu-calculator/')

  it('renders both steps, in order', () => {
    const cache = html.indexOf('KV cache')
    const gpu = html.indexOf('GPU configuration')
    expect(cache).toBeGreaterThan(-1)
    expect(gpu).toBeGreaterThan(cache)
  })

  it('opens the cache step and shows its form', () => {
    expect(html).toContain('id="step-1-trigger"')
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('Model source')
    expect(html).toContain('id="kv-model-id"')
    expect(html).toContain('id="context-length"')
    expect(html).toContain('id="sequence-count"')
    expect(html).toContain('id="kv-cache-dtype"')
  })

  it('keeps the hardware step closed and disabled', () => {
    expect(html).toContain('id="step-2-trigger"')
    expect(html).toContain('disabled=""')
    expect(html).toContain('Size the KV cache in step 1')
    // The content is unmounted, so none of its fields are in the markup.
    expect(html).not.toContain('id="inference-headroom"')
    expect(html).not.toContain('id="inference-max-gpus"')
    expect(html).not.toContain('id="inference-weight-format"')
  })

  it('shows the cache panel reading the config rather than a wrong number', () => {
    expect(html).toContain('Reading the model config')
  })

  it('carries the questions the page answers', () => {
    expect(html).toContain('Questions about inference hardware')
    for (const item of INFERENCE_FAQ) {
      expect(html).toContain(item.question)
    }
  })

  it('names the weight format on offer', () => {
    // The weight format picker lives in the closed step, so the page names it
    // in its own description instead.
    expect(html).toContain('weight format the checkpoint publishes')
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

describe('KvCacheCalculator at the first paint', () => {
  const html = renderPage(createElement(KvCacheCalculator), '/tools/kv-cache-calculator/')

  it('renders one page with the model source and the cache fields', () => {
    expect(html).toContain('Model source')
    expect(html).toContain('id="kv-model-id"')
    expect(html).toContain('id="context-length"')
    expect(html).toContain('id="sequence-count"')
    expect(html).toContain('id="kv-cache-dtype"')
  })

  it('offers the three model sources', () => {
    expect(html).toContain('Model id')
    expect(html).toContain('Paste config')
    expect(html).toContain('Enter numbers')
  })

  it('has no step section, because it answers one question', () => {
    expect(html).not.toContain('id="step-1-trigger"')
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
