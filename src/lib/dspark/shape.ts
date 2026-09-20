import { readNumber, unwrapConfig } from '../model-config'
import type { RawConfig } from '../model-config'
import { attentionParamsPerLayer } from '../model-shape'
import { readMoeShapeFields } from '../model-shape/moe'

import type { DsparkTargetShape } from './types'

/**
 * Reads the target shape a DSpark run has to be costed against.
 *
 * DSpark attaches a drafter to a frozen target and shares that target's
 * embedding and language model head, so the target's width sets the cache width,
 * the draft's parameter count, and how much context each draft block attends to.
 * Everything the estimator needs is derivable from a config.
 *
 * Returns null when the config has no hidden size or depth, which means it is
 * not a transformer config and there is nothing to cost out.
 *
 * The expert bank is read by the shared reader, so a mixture of experts target
 * is costed with the same expert arithmetic the REAP calculator uses. A dense
 * target is handled here, and is the common case: the published DSpark drafters
 * attach to dense Qwen3 and Gemma targets.
 */
export function detectDsparkShape(config: RawConfig): DsparkTargetShape | null {
  const { inner, outer } = unwrapConfig(config)
  const notes: string[] = []
  let bestEffort = false

  const hiddenSize = readNumber(inner, 'hidden_size') ?? readNumber(outer, 'hidden_size') ?? 0
  if (hiddenSize < 1) return null

  const numLayers = readNumber(inner, 'num_hidden_layers') ?? 0
  if (numLayers < 1) return null

  const modelType =
    typeof inner.model_type === 'string'
      ? inner.model_type
      : typeof outer.model_type === 'string'
        ? outer.model_type
        : 'unknown'

  const intermediateSize = readNumber(inner, 'intermediate_size') ?? 4 * hiddenSize
  if (readNumber(inner, 'intermediate_size') === undefined) {
    notes.push(
      `The config has no intermediate_size, so four times the hidden width of ${hiddenSize.toLocaleString('en-US')} is assumed for the feed forward block.`,
    )
  }

  const vocabSize = readNumber(inner, 'vocab_size') ?? readNumber(outer, 'vocab_size') ?? 0
  if (vocabSize < 1) {
    bestEffort = true
    notes.push(
      'The config has no vocab_size, so the Markov head and the shared embedding are understated. Both scale with the vocabulary.',
    )
  }

  const attention = attentionParamsPerLayer(config)

  // --- Expert bank, when the target has one -------------------------------

  const moe = readMoeShapeFields(config)
  const routedExperts = moe?.routedExperts ?? 0
  const expertsPerToken = moe?.expertsPerToken ?? 0
  const moeIntermediateSize = moe?.moeIntermediateSize ?? intermediateSize
  const moeLayers = moe?.moeLayers ?? 0
  const denseLayers = numLayers - moeLayers
  if (moe) notes.push(...moe.notes)

  // SwiGLU experts carry three projections: gate, up, and down.
  const paramsPerExpert = 3 * hiddenSize * moeIntermediateSize

  const attentionParams = numLayers * attention
  const denseFfnParams = denseLayers * 3 * hiddenSize * intermediateSize
  const routedExpertParams = moeLayers * routedExperts * paramsPerExpert
  const embedParams = vocabSize * hiddenSize

  const totalParams = attentionParams + denseFfnParams + routedExpertParams + embedParams

  // The router is a single small matmul per block, well under one percent of a
  // block here, so it is left out to keep the top-k's effect on compute visible.
  const activeParamsPerToken =
    attentionParams + denseFfnParams + moeLayers * expertsPerToken * paramsPerExpert

  return {
    modelType,
    hiddenSize,
    numLayers,
    intermediateSize,
    vocabSize,
    attentionParamsPerLayer: attention,
    routedExperts,
    expertsPerToken,
    moeIntermediateSize,
    moeLayers,
    totalParams,
    activeParamsPerToken,
    bestEffort,
    notes,
  }
}
