import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { findGpu, findStorage } from '@/lib/hardware'
import type { GpuSpec } from '@/lib/hardware'
import {
  DEFAULT_BLOCK_SIZE,
  DEFAULT_DRAFT_LAYERS,
  DEFAULT_MARKOV_RANK,
  DEFAULT_MFU,
  DEFAULT_MICRO_BATCH,
  DEFAULT_NUM_ANCHORS,
  DEFAULT_OVERHEAD_FACTOR,
  DEFAULT_SETUP_SECONDS,
  DEFAULT_TARGET_LAYERS,
  detectDsparkShape,
  estimateDspark,
  type DsparkInputs,
  type DsparkResult,
} from '@/lib/dspark'
import { parseConfigText } from '@/lib/model-config'
import { readConfigFixtureText } from '@/test/fixtures'

import DsparkResults from './DsparkResults'
import type { DsparkShapeState } from './useDsparkShape'

const RTX_5090 = findGpu('rtx-5090') as GpuSpec

function inputs(overrides: Partial<DsparkInputs> = {}): DsparkInputs {
  return {
    numTargetLayers: DEFAULT_TARGET_LAYERS,
    trainingTokens: 1_237_000_000,
    epochs: 10,
    numAnchors: DEFAULT_NUM_ANCHORS,
    blockSize: DEFAULT_BLOCK_SIZE,
    numDraftLayers: DEFAULT_DRAFT_LAYERS,
    markovRank: DEFAULT_MARKOV_RANK,
    sequenceLength: 4096,
    dataMode: 'offline',
    gpu: RTX_5090,
    storage: findStorage('pcie5-host-ram')!,
    gpuCount: 1,
    mfu: DEFAULT_MFU,
    overheadFactor: DEFAULT_OVERHEAD_FACTOR,
    setupSeconds: DEFAULT_SETUP_SECONDS,
    microBatchSize: DEFAULT_MICRO_BATCH,
    ...overrides,
  }
}

/**
 * Renders the results panel the way the page does, so the wiring from a pasted
 * config through the estimator to the figures on screen is covered end to end.
 */
function render(configText: string, overrides: Partial<DsparkInputs> = {}): string {
  const config = parseConfigText(configText)
  const shape = detectDsparkShape(config)
  const dsparkInputs = inputs(overrides)
  const result: DsparkResult | null = shape ? estimateDspark(shape, dsparkInputs) : null
  const shapeState: DsparkShapeState = {
    status: shape ? 'ready' : 'error',
    shape,
    config,
    configUrl: null,
    error: shape ? null : 'That config is not a decoder.',
  }
  return renderToString(
    createElement(DsparkResults, {
      result,
      shapeState,
      gpu: dsparkInputs.gpu,
      computeError: null,
    }),
  )
}

/**
 * A real published target, so the panel is exercised against a config the tool
 * would actually be pointed at.
 */
const QWEN3_4B = JSON.stringify({
  model_type: 'qwen3',
  hidden_size: 2560,
  num_hidden_layers: 36,
  intermediate_size: 9728,
  vocab_size: 151936,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  head_dim: 128,
  tie_word_embeddings: true,
})

/**
 * Rendered HTML with React's text-node separators removed.
 *
 * A phrase that spans an interpolated expression is broken up by comment
 * markers in the output, so a multi-word assertion has to read the text the way
 * a browser would rather than the way React emitted it.
 */
function visibleText(html: string): string {
  return html.replace(/<!-- -->/g, '')
}

describe('DsparkResults with a pasted target config', () => {
  it('renders a duration', () => {
    const html = render(QWEN3_4B)
    expect(html).toContain('Estimated training run')
    expect(html).toMatch(/\d+(\.\d+)?\s*(seconds|minutes|hours|days)/)
  })

  it('renders the cache size and what it is made of', () => {
    const html = render(QWEN3_4B)
    expect(html).toContain('How big is the target cache?')
    expect(html).toContain('of target hidden states')
    expect(html).toMatch(/\d+(\.\d+)?\s*(GiB|TiB)/)
    expect(html).toContain('captured target layers')
  })

  it('renders a duration section that names the bound', () => {
    const html = render(QWEN3_4B)
    expect(html).toContain('How long does the run take?')
    expect(visibleText(html)).toMatch(
      /limited by the (GPU that does the arithmetic|storage that feeds the target cache back)/,
    )
  })

  it('renders a memory verdict', () => {
    const html = render(QWEN3_4B)
    expect(html).toContain('Does it fit in VRAM?')
    expect(html).toContain('RTX 5090')
    expect(html).toMatch(/Fits|Needs offline capture|Needs more GPUs|Needs offloading/)
  })

  it('renders the drafter cost', () => {
    const html = render(QWEN3_4B)
    expect(html).toContain('What does the drafter cost?')
    expect(html).toContain('percent of the target')
    expect(html).toContain('Markov head')
  })

  it('says the target is not resident offline and is online', () => {
    const offline = render(QWEN3_4B, { dataMode: 'offline' })
    const online = render(QWEN3_4B, { dataMode: 'online' })
    expect(offline).toContain('the target out of VRAM while the drafter trains')
    expect(online).toContain('keeps the')
    expect(online).toContain('target in VRAM for the whole run')
    // Online writes no cache, so the cache section says so instead of a size.
    expect(online).toContain('No target cache, because the run captures the target online')
  })

  it('recommends offline capture when only the resident target breaks the budget', () => {
    // 12 GiB holds the drafter but not the drafter plus a 4B target.
    const gpu = findGpu('rtx-5070') as GpuSpec
    expect(render(QWEN3_4B, { gpu, dataMode: 'offline' })).toContain('Fits')
    expect(render(QWEN3_4B, { gpu, dataMode: 'online' })).toContain('Needs offline capture')
  })

  it('becomes bound by the cache read on a slow disk', () => {
    // A single card is fast enough that a network share is not the bottleneck,
    // but eight are, which is the case the bound warning exists for.
    const html = render(QWEN3_4B, {
      storage: findStorage('network-10gbe')!,
      gpuCount: 8,
    })
    expect(visibleText(html)).toContain('storage that feeds the target cache back')
    expect(visibleText(html)).toContain('of target cache reads')
  })

  it('stays bound by compute when the cache sits in host memory', () => {
    const html = render(QWEN3_4B, { gpuCount: 8 })
    expect(visibleText(html)).toContain('the GPU that does the arithmetic')
  })

  it('renders a mixture of experts target without falling over', () => {
    // The shared reader is what detects the expert bank, so this checks the
    // wiring rather than the expert arithmetic, which the engine tests cover.
    const html = render(readConfigFixtureText('glm-5-3'))
    expect(html).toContain('Estimated training run')
    expect(html).toMatch(/Fits|Needs offline capture|Needs more GPUs|Needs offloading/)
  })
})

describe('DsparkResults in the idle, error and compute error states', () => {
  const idleState: DsparkShapeState = {
    status: 'idle',
    shape: null,
    config: null,
    configUrl: null,
    error: null,
  }

  it('prompts for input before anything is entered', () => {
    const html = renderToString(
      createElement(DsparkResults, {
        result: null,
        shapeState: idleState,
        gpu: RTX_5090,
        computeError: null,
      }),
    )
    expect(html).toContain('Enter a target')
  })

  it('surfaces a config that is not a decoder', () => {
    const html = renderToString(
      createElement(DsparkResults, {
        result: null,
        shapeState: {
          ...idleState,
          status: 'error',
          error: 'That config does not describe a decoder.',
        },
        gpu: RTX_5090,
        computeError: null,
      }),
    )
    expect(html).toContain('Could not read the target config')
    expect(html).toContain('That config does not describe a decoder.')
  })

  it('surfaces an input error from the estimator', () => {
    const html = renderToString(
      createElement(DsparkResults, {
        result: null,
        shapeState: {
          ...idleState,
          status: 'ready',
          shape: detectDsparkShape(JSON.parse(QWEN3_4B)),
        },
        gpu: RTX_5090,
        computeError: 'Epochs must be a whole number of 1 or more.',
      }),
    )
    expect(html).toContain('Those inputs cannot be costed')
    expect(html).toContain('Epochs must be a whole number of 1 or more.')
  })
})
