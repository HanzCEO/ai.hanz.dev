import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { estimateInference } from '@/lib/inference'
import { formatBytes } from '@/lib/format'
import { computeKvCache } from '@/lib/kvcache'
import { detectModelShape } from '@/lib/model-shape'
import { loadConfigFixture } from '@/test/fixtures'

import { describeCacheStep } from './cacheSummary'
import InferenceLayers from './InferenceLayers'

const QWEN3_8B = loadConfigFixture('qwen3-8b')

/**
 * Renders the two steps in a given state.
 *
 * The component takes the resolved state and the two forms as children, so it
 * can be rendered here without a fetch and without the page's own state.
 */
function render(options: {
  cacheOpen: boolean
  gpuOpen: boolean
  gpuReady: boolean
  cacheSummary?: string
}): string {
  return renderToString(
    createElement(InferenceLayers, {
      cacheOpen: options.cacheOpen,
      gpuOpen: options.gpuOpen,
      onCacheOpenChange: () => {},
      onGpuOpenChange: () => {},
      gpuReady: options.gpuReady,
      cacheSummary: options.cacheSummary ?? 'The cache summary',
      cacheForm: createElement('div', { 'data-testid': 'cache-form' }, 'The cache form'),
      cacheResults: createElement('div', null, 'The cache result'),
      gpuForm: createElement('div', { 'data-testid': 'gpu-form' }, 'The hardware form'),
      gpuResults: createElement('div', null, 'The GPU answer'),
    }),
  )
}

describe('InferenceLayers', () => {
  it('names both steps, in order', () => {
    const html = render({ cacheOpen: true, gpuOpen: false, gpuReady: false })
    const cache = html.indexOf('KV cache')
    const gpu = html.indexOf('GPU configuration')
    expect(cache).toBeGreaterThan(-1)
    expect(gpu).toBeGreaterThan(-1)
    expect(cache).toBeLessThan(gpu)
  })

  it('numbers the steps', () => {
    const html = render({ cacheOpen: true, gpuOpen: false, gpuReady: false })
    expect(html).toContain('>1</span>')
    expect(html).toContain('>2</span>')
  })

  it('opens the first step and disables the second while the cache has no size', () => {
    const html = render({ cacheOpen: true, gpuOpen: false, gpuReady: false })
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('aria-expanded="false"')
    expect(html).toContain('disabled=""')
    expect(html).toContain('The cache form')
    expect(html).not.toContain('The hardware form')
  })

  it('teases the second step while it is disabled', () => {
    const html = render({ cacheOpen: true, gpuOpen: false, gpuReady: false })
    expect(html).toContain('Size the KV cache in step 1')
  })

  it('unmounts the first form and shows its summary once the step is closed', () => {
    const html = render({
      cacheOpen: false,
      gpuOpen: true,
      gpuReady: true,
      cacheSummary: 'Qwen/Qwen3-8B holds a cache of 1.2 GiB in BF16',
    })
    expect(html).not.toContain('The cache form')
    expect(html).toContain('Qwen/Qwen3-8B holds a cache of 1.2 GiB in BF16')
    expect(html).toContain('The hardware form')
    expect(html).toContain('The GPU answer')
  })

  it('lets both steps be open at the same time', () => {
    const html = render({ cacheOpen: true, gpuOpen: true, gpuReady: true })
    expect(html).toContain('The cache form')
    expect(html).toContain('The hardware form')
  })

  it('gives the second step no summary once it is open', () => {
    const html = render({ cacheOpen: false, gpuOpen: true, gpuReady: true })
    expect(html).not.toContain('Size the KV cache in step 1')
  })

  it('uses no em dash or en dash', () => {
    const pages = [
      render({ cacheOpen: true, gpuOpen: false, gpuReady: false }),
      render({ cacheOpen: false, gpuOpen: true, gpuReady: true }),
      render({ cacheOpen: true, gpuOpen: true, gpuReady: true }),
    ]
    for (const html of pages) {
      expect(html).not.toContain('\u2014')
      expect(html).not.toContain('\u2013')
    }
  })
})

describe('the collapsed cache summary', () => {
  const config = loadConfigFixture('qwen3-8b')
  /** An MLA model, so it holds a separate indexer cache with its own dtype. */
  const indexed = loadConfigFixture('glm-5-3')

  function cacheOf(source: typeof config, kvCacheDtype: string, indexerDtype: string) {
    return computeKvCache(source, {
      contextLength: 32768,
      sequenceCount: 4,
      kvCacheDtype: kvCacheDtype as never,
      indexerDtype: indexerDtype as never,
    })
  }

  it('names the model, the context, the sequences, the dtype and the size', () => {
    const result = cacheOf(config, 'FP8_E4M3', 'FP8_E4M3')
    const summary = describeCacheStep(result, 'Qwen/Qwen3-8B', 'FP8_E4M3', 'FP8_E4M3')
    expect(summary).toContain('Qwen/Qwen3-8B')
    expect(summary).toContain('32,768 tokens')
    expect(summary).toContain('4 sequences')
    expect(summary).toContain('FP8_E4M3')
    expect(summary).toContain(formatBytes(result.totalBytes).text)
    expect(summary).toContain('Reopen step 1')
  })

  it('says sequence rather than sequences for one', () => {
    const result = computeKvCache(config, {
      contextLength: 8192,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })
    const summary = describeCacheStep(result, 'Qwen/Qwen3-8B', 'BF16')
    expect(summary).toContain('1 sequence ')
    expect(summary).not.toContain('1 sequences')
  })

  it('names one dtype for a model with no indexer cache', () => {
    const result = cacheOf(config, 'FP8_E4M3', 'BF16')
    expect(result.indexerDtypeSupport).toBeNull()
    const summary = describeCacheStep(result, 'Qwen/Qwen3-8B', 'FP8_E4M3', 'BF16')
    expect(summary).toContain('in FP8_E4M3.')
    // The indexer dtype is meaningless for this model, so it is not named.
    expect(summary).not.toContain('indexer')
  })

  it('names one dtype when a model with an indexer uses the same dtype for both', () => {
    const result = cacheOf(indexed, 'FP8_E4M3', 'FP8_E4M3')
    expect(result.indexerDtypeSupport).not.toBeNull()
    const summary = describeCacheStep(result, 'zai-org/GLM-5.3', 'FP8_E4M3', 'FP8_E4M3')
    expect(summary).toContain('in FP8_E4M3.')
    expect(summary).not.toContain('indexer')
  })

  it('names both dtypes when a model with an indexer uses a mixed pair', () => {
    const result = cacheOf(indexed, 'FP8_E4M3', 'FP4')
    const summary = describeCacheStep(result, 'zai-org/GLM-5.3', 'FP8_E4M3', 'FP4')
    // Naming only one would misreport the pair, so both are named.
    expect(summary).toContain('The attention cache is in FP8_E4M3')
    expect(summary).toContain('the indexer cache is in FP4')
  })

  it('keeps every sentence to 20 words or fewer', () => {
    const summaries = [
      describeCacheStep(cacheOf(config, 'FP8_E4M3', 'BF16'), 'Qwen/Qwen3-8B', 'FP8_E4M3', 'BF16'),
      describeCacheStep(cacheOf(indexed, 'FP8_E4M3', 'FP4'), 'zai-org/GLM-5.3', 'FP8_E4M3', 'FP4'),
    ]
    for (const summary of summaries) {
      const sentences = summary.split(/(?<=[.!?])\s+/).filter((part) => part !== '')
      for (const sentence of sentences) {
        const words = sentence.split(/\s+/).filter((word) => word !== '').length
        expect(words, `${words} words: ${sentence}`).toBeLessThanOrEqual(20)
      }
    }
  })

  it('uses no em dash or en dash', () => {
    const summaries = [
      describeCacheStep(cacheOf(config, 'BF16', 'BF16'), 'Qwen/Qwen3-8B', 'BF16', 'BF16'),
      describeCacheStep(cacheOf(indexed, 'FP8_E4M3', 'FP4'), 'zai-org/GLM-5.3', 'FP8_E4M3', 'FP4'),
    ]
    for (const summary of summaries) {
      expect(summary).not.toContain('\u2014')
      expect(summary).not.toContain('\u2013')
    }
  })
})

describe('the two steps agree on the cache', () => {
  /**
   * The hardware step must cost the same cache the cache step reports. Both
   * read the same engine, so the two numbers are the same bytes, not a rounded
   * pair that happens to look close.
   */
  it('reports the same cache bytes for the same inputs and dtypes', () => {
    const shape = detectModelShape(QWEN3_8B)
    expect(shape).not.toBeNull()

    const cases = [
      { contextLength: 8192, sequences: 1, dtype: 'BF16' as const },
      { contextLength: 32768, sequences: 4, dtype: 'BF16' as const },
      { contextLength: 131072, sequences: 16, dtype: 'FP8_E4M3' as const },
    ]

    for (const item of cases) {
      const cacheStep = computeKvCache(QWEN3_8B, {
        contextLength: item.contextLength,
        sequenceCount: item.sequences,
        kvCacheDtype: item.dtype,
        indexerDtype: item.dtype,
      })
      const hardwareStep = estimateInference(shape!, {
        config: QWEN3_8B,
        weightFormat: 'BF16',
        contextLength: item.contextLength,
        sequences: item.sequences,
        headroom: 0.1,
        maxGpus: 8,
        kvCacheDtype: item.dtype,
        indexerDtype: item.dtype,
      })

      expect(hardwareStep.kvCacheBytes, `context ${item.contextLength}`).toBe(cacheStep.totalBytes)
      expect(hardwareStep.kvBytesPerToken).toBe(cacheStep.bytesPerToken)
    }
  })

  it('reports a larger cache for a longer context in both steps', () => {
    const shape = detectModelShape(QWEN3_8B)!
    const short = computeKvCache(QWEN3_8B, {
      contextLength: 8192,
      sequenceCount: 1,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })
    const long = estimateInference(shape, {
      config: QWEN3_8B,
      weightFormat: 'BF16',
      contextLength: 16384,
      sequences: 1,
      headroom: 0.1,
      maxGpus: 8,
      kvCacheDtype: 'BF16',
      indexerDtype: 'BF16',
    })
    expect(long.kvCacheBytes).toBe(short.totalBytes * 2)
  })
})
