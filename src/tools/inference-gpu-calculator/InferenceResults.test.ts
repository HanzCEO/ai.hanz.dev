import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { estimateInference, type InferenceInputs, type InferenceResult } from '@/lib/inference'
import type { RawConfig } from '@/lib/model-config'
import { detectModelShape } from '@/lib/model-shape'
import type { ConfigSourceState } from '@/lib/use-config-source'
import { loadConfigFixture } from '@/test/fixtures'

import InferenceResults from './InferenceResults'

const QWEN3_8B = loadConfigFixture('qwen3-8b')

/**
 * A 70B class dense model, so a single consumer card cannot hold it and the
 * ranking has to divide the weights across several.
 */
const LARGE: RawConfig = {
  model_type: 'llama',
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  head_dim: 128,
  intermediate_size: 28672,
  vocab_size: 128256,
  tie_word_embeddings: false,
}

function inputs(config: RawConfig, overrides: Partial<InferenceInputs> = {}): InferenceInputs {
  return {
    config,
    precision: 'BF16',
    contextLength: 8192,
    sequences: 1,
    headroom: 0.1,
    maxGpus: 8,
    ...overrides,
  }
}

function readyState(config: RawConfig): ConfigSourceState {
  return { status: 'ready', config, url: null, error: null }
}

/**
 * Renders the panel the way the page does, so the wiring from a config through
 * the estimator to the figures on screen is covered end to end.
 */
function render(config: RawConfig, overrides: Partial<InferenceInputs> = {}): string {
  const shape = detectModelShape(config)
  const result: InferenceResult | null = shape
    ? estimateInference(shape, inputs(config, overrides))
    : null
  return renderToString(
    createElement(InferenceResults, {
      result,
      configState: readyState(config),
      shape,
      computeError: null,
      modelLabel: 'Qwen/Qwen3-8B',
    }),
  )
}

/** Rendered HTML with React's text-node separators removed. */
function visibleText(html: string): string {
  return html.replace(/<!-- -->/g, '')
}

describe('InferenceResults with a model that fits one card', () => {
  it('names the recommended card and count', () => {
    const html = visibleText(render(QWEN3_8B, { gpuFilter: ['rtx-4090'] }))
    expect(html).toContain('Recommended configuration')
    expect(html).toContain('1 x RTX 4090')
    expect(html).toContain('One card')
  })

  it('states the answer in one self-contained sentence', () => {
    const html = visibleText(render(QWEN3_8B, { gpuFilter: ['rtx-4090'] }))
    expect(html).toContain('Qwen/Qwen3-8B')
    expect(html).toContain('needs')
    expect(html).toContain('1 x RTX 4090')
    expect(html).toContain('for')
  })

  it('renders the memory terms the answer is made of', () => {
    const html = render(QWEN3_8B, { gpuFilter: ['rtx-4090'] })
    expect(html).toContain('Resident weights')
    expect(html).toContain('KV cache')
    expect(html).toContain('Activation buffer')
    expect(html).toContain('Framework reserve')
    expect(html).toContain('Total needed')
    expect(html).toMatch(/\d+(\.\d+)?\s*(MiB|GiB)/)
  })

  it('names the cache dtype the answer was costed with', () => {
    const html = render(QWEN3_8B, { gpuFilter: ['rtx-4090'] })
    expect(html).toContain('Cache dtype')
    expect(html).toContain('BF16')
  })

  it('renders the room left and the decode rate', () => {
    const html = render(QWEN3_8B, { gpuFilter: ['rtx-4090'] })
    expect(html).toContain('How much room is left?')
    expect(html).toContain('free on one card')
    expect(html).toContain('How fast does it go?')
    expect(html).toMatch(/tokens\/s/)
  })

  it('lists every configuration that fits and marks the recommended one', () => {
    const html = render(QWEN3_8B, { gpuFilter: ['rtx-4090', 'h100'] })
    expect(html).toContain('Every configuration that fits')
    expect(html).toContain('1 x RTX 4090')
    expect(html).toContain('1 x H100 SXM')
    expect(html).toContain('Recommended')
  })

  it('renders the breakdown panels', () => {
    const html = render(QWEN3_8B, { gpuFilter: ['rtx-4090'] })
    expect(html).toContain('How this was calculated')
    expect(html).toContain('Memory in detail')
    expect(html).toContain('What one card holds')
    expect(html).toContain('Assumptions and caveats')
  })

  it('costs a narrower cache dtype on a smaller card than BF16 needs', () => {
    const wide = detectModelShape(QWEN3_8B)
    const bf16 = wide
      ? estimateInference(wide, inputs(QWEN3_8B, { contextLength: 32768, sequences: 4 }))
      : null
    const fp8 = wide
      ? estimateInference(
          wide,
          inputs(QWEN3_8B, {
            contextLength: 32768,
            sequences: 4,
            kvCacheDtype: 'FP8_E4M3',
            indexerDtype: 'FP8_E4M3',
          }),
        )
      : null

    expect(bf16).not.toBeNull()
    expect(fp8).not.toBeNull()
    expect(fp8!.kvCacheBytes).toBeLessThan(bf16!.kvCacheBytes)
    expect(fp8!.totalBytes).toBeLessThan(bf16!.totalBytes)
    expect(fp8!.weightsBytes).toBe(bf16!.weightsBytes)
  })
})

describe('InferenceResults with a model that needs several cards', () => {
  it('names the multi card verdict', () => {
    const html = visibleText(render(LARGE, { gpuFilter: ['rtx-4090'] }))
    expect(html).toContain('Several cards')
    expect(html).toMatch(/\d+ x RTX 4090/)
    expect(html).toContain('does not halve the footprint')
  })
})

describe('InferenceResults with a model that does not fit', () => {
  it('renders an alert that names the shortfall', () => {
    const html = visibleText(render(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 2 }))
    expect(html).toContain('No configuration in this list holds the model')
    expect(html).toContain('2 cards')
    expect(html).toContain('The nearest configuration is')
    expect(html).toContain('Nothing fits')
  })

  it('names the three options that remain', () => {
    const html = visibleText(render(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 2 }))
    expect(html).toContain('Quantize the weights')
    expect(html).toContain('Offload part of the model')
    expect(html).toContain('raise the maximum GPU count')
  })

  it('lists no configuration table, because none fits', () => {
    const html = render(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 2 })
    expect(html).not.toContain('Every configuration that fits')
  })
})

describe('InferenceResults before a config arrives', () => {
  it('asks for a model while idle', () => {
    const html = renderToString(
      createElement(InferenceResults, {
        result: null,
        configState: { status: 'idle', config: null, url: null, error: null },
        shape: null,
        computeError: null,
      }),
    )
    expect(html).toContain('Enter a model')
  })

  it('says it is reading the config while loading', () => {
    const html = renderToString(
      createElement(InferenceResults, {
        result: null,
        configState: { status: 'loading', config: null, url: null, error: null },
        shape: null,
        computeError: null,
      }),
    )
    expect(html).toContain('Reading the model config')
    expect(html).toContain('role="status"')
  })

  it('reports a config it cannot read', () => {
    const html = renderToString(
      createElement(InferenceResults, {
        result: null,
        configState: {
          status: 'error',
          config: null,
          url: null,
          error: new Error('The calculator cannot read that model config.') as never,
        },
        shape: null,
        computeError: null,
      }),
    )
    expect(html).toContain('The calculator cannot read that config')
    expect(html).toContain('The calculator cannot read that model config.')
  })

  it('reports a config that describes no transformer', () => {
    const html = renderToString(
      createElement(InferenceResults, {
        result: null,
        configState: readyState({ model_type: 'gateway' }),
        shape: null,
        computeError: null,
      }),
    )
    expect(html).toContain('The calculator cannot size that config')
    expect(html).toContain('hidden size')
  })

  it('reports inputs the estimator rejected', () => {
    const html = renderToString(
      createElement(InferenceResults, {
        result: null,
        configState: readyState(QWEN3_8B),
        shape: detectModelShape(QWEN3_8B),
        computeError: 'The VRAM headroom must be at least 0 percent and less than 100 percent.',
      }),
    )
    expect(html).toContain('Those inputs cannot be costed')
    expect(html).toContain('VRAM headroom')
  })
})

describe('InferenceResults copy', () => {
  it('uses no em dash or en dash anywhere it renders', () => {
    const pages = [
      render(QWEN3_8B, { gpuFilter: ['rtx-4090'] }),
      render(LARGE, { gpuFilter: ['rtx-4090'] }),
      render(LARGE, { gpuFilter: ['rtx-4090'], maxGpus: 2 }),
    ]
    for (const html of pages) {
      expect(html).not.toContain('\u2014')
      expect(html).not.toContain('\u2013')
    }
  })

  it('uses no contraction anywhere it renders', () => {
    const html = visibleText(render(QWEN3_8B, { gpuFilter: ['rtx-4090'] }))
    for (const contraction of ['doesn\u2019t', "doesn't", "isn't", "it's"]) {
      expect(html).not.toContain(contraction)
    }
  })
})
