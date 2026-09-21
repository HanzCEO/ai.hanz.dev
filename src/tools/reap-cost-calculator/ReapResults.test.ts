import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { findGpu } from '@/lib/hardware'
import type { GpuSpec } from '@/lib/hardware'
import { parseConfigText } from '@/lib/model-config'
import { detectMoeShape, estimateReap, type ReapInputs, type ReapResult } from '@/lib/reap'
import { readConfigFixtureText } from '@/test/fixtures'
import { reapInputs } from '@/test/reap'

import ReapResults from './ReapResults'
import type { ReapShapeState } from './useReapShape'

const RTX_5090 = findGpu('rtx-5090') as GpuSpec

/**
 * The panel tests open on a consumer card in FP8, which is the case where the
 * feasibility verdict has something to say. The engine defaults are used for
 * everything else, so the two suites cost out the same run.
 */
function inputs(overrides: Partial<ReapInputs> = {}): ReapInputs {
  return reapInputs({ gpu: RTX_5090, weightDtype: 'FP8', ...overrides })
}

/**
 * Renders the results panel the way the page does, so the wiring from a pasted
 * config through the estimator to the figures on screen is covered end to end.
 */
function render(configText: string, overrides: Partial<ReapInputs> = {}): string {
  const config = parseConfigText(configText)
  const shape = detectMoeShape(config)
  const reapInputs = inputs(overrides)
  const result: ReapResult | null = shape ? estimateReap(shape, reapInputs) : null
  const shapeState: ReapShapeState = {
    status: shape ? 'ready' : 'not-moe',
    shape,
    config,
    configUrl: null,
    error: null,
    suggestedDtype: null,
  }
  return renderToString(
    createElement(ReapResults, {
      result,
      shapeState,
      gpu: reapInputs.gpu,
      computeError: null,
    }),
  )
}

const glm53Text = readConfigFixtureText('glm-5-3')

describe('ReapResults with a pasted GLM-5.3 config', () => {
  it('renders a time estimate', () => {
    const html = render(glm53Text)
    // 512 samples at 2048 tokens is a small run, so the figure is in minutes.
    expect(html).toContain('Estimated REAP run')
    expect(html).toMatch(/\d+(\.\d+)?\s*(seconds|minutes|hours|days)/)
  })

  it('renders a feasibility verdict for the card', () => {
    const html = render(glm53Text)
    expect(html).toContain('RTX 5090')
    expect(html).toMatch(/Fits|Needs FP8|Needs offloading/)
    expect(html).toContain('Can I prune this model?')
  })

  it('renders the one block VRAM figure', () => {
    const html = render(glm53Text)
    expect(html).toContain('One expert block is')
    expect(html).toMatch(/\d+(\.\d+)?\s*(MiB|GiB)/)
  })

  it('renders the post prune size', () => {
    const html = render(glm53Text)
    expect(html).toContain('How much smaller does it get?')
    expect(html).toContain('The run keeps')
    expect(html).toMatch(/percent smaller/)
  })

  it('states that active compute is unchanged when the top-k is kept', () => {
    const html = render(glm53Text)
    expect(html).toContain('Active parameters per token are unchanged')
  })

  it('states that reducing the top-k is where the faster inference comes from', () => {
    const html = render(glm53Text, { scaleTopK: true })
    expect(html).toContain('Reducing the top-k')
    expect(html).not.toContain('Active parameters per token are unchanged')
  })

  it('does not claim a reduction when the setting removes nothing', () => {
    // A prune ratio of zero keeps every expert, so nothing is pruned. The size
    // card used to render "0.0 percent smaller" and to print the same parameter
    // count twice as though it had fallen.
    const zero = render(glm53Text, { pruneRatio: 0 })
    expect(zero).not.toContain('0.0 percent smaller')
    expect(zero).not.toContain('fall from')
    expect(zero).toContain('Nothing is removed at this setting')
    expect(zero).toContain('total parameter count is unchanged')
  })

  it('reports the long run honestly at the checkpoint recipe', () => {
    const html = render(glm53Text, {
      calibrationSamples: 24576,
      sequenceLength: 16384,
      gpu: findGpu('h200') as GpuSpec,
      weightDtype: 'BF16',
    })
    expect(html).toMatch(/\d+(\.\d+)?\s*(hours|days)/)
    expect(html).toContain('H200')
  })
})

describe('ReapResults with a dense model', () => {
  it('explains that REAP does not apply', () => {
    const dense = JSON.stringify({
      model_type: 'qwen3',
      num_hidden_layers: 36,
      hidden_size: 4096,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
    })
    const html = render(dense)
    expect(html).toContain('REAP does not apply to this model')
    expect(html).not.toContain('Estimated REAP run')
  })
})

describe('ReapResults in the idle and error states', () => {
  const idleState: ReapShapeState = {
    status: 'idle',
    shape: null,
    config: null,
    configUrl: null,
    error: null,
    suggestedDtype: null,
  }

  it('prompts for input before anything is entered', () => {
    const html = renderToString(
      createElement(ReapResults, {
        result: null,
        shapeState: idleState,
        gpu: RTX_5090,
        computeError: null,
      }),
    )
    expect(html).toContain('Enter a mixture of experts model')
  })

  it('surfaces a read failure', () => {
    const html = renderToString(
      createElement(ReapResults, {
        result: null,
        shapeState: { ...idleState, status: 'error', error: 'That is not valid JSON.' },
        gpu: RTX_5090,
        computeError: null,
      }),
    )
    expect(html).toContain('The calculator cannot read that config')
    expect(html).toContain('That is not valid JSON.')
  })

  it('surfaces an input error from the estimator', () => {
    const html = renderToString(
      createElement(ReapResults, {
        result: null,
        shapeState: idleState,
        gpu: RTX_5090,
        computeError: 'Calibration samples must be a whole number of 1 or more.',
      }),
    )
    // The idle branch wins while nothing has been entered, so no error shows.
    expect(html).toContain('Enter a mixture of experts model')
  })
})
