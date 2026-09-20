import { attentionParamsPerLayer } from '../model-shape'
import { readMoeShapeFields } from '../model-shape/moe'
import type { RawConfig } from '../model-config'

import type { MoeShape } from './types'

/**
 * Reads the MoE shape out of a config, or returns null when the model has no
 * expert bank.
 *
 * REAP only applies to a sparsely activated mixture of experts. A dense model
 * has nothing to prune, so the caller is told so rather than being handed a
 * meaningless number.
 *
 * The reading of the config itself is shared with every other calculator that
 * sizes a model; what is specific to REAP is the parameter accounting below.
 */
export function detectMoeShape(config: RawConfig): MoeShape | null {
  const fields = readMoeShapeFields(config)
  if (!fields) return null

  const {
    modelType,
    hiddenSize,
    numLayers,
    intermediateSize,
    moeIntermediateSize,
    routedExperts,
    expertsPerToken,
    sharedExperts,
    sharedExpertIntermediate,
    moeLayers,
    denseLayers,
    vocabSize,
    tiedEmbeddings,
    notes,
  } = fields

  // A missing router top-k is the one inference that moves the headline number,
  // so it is flagged rather than only noted.
  const bestEffort = expertsPerToken === undefined

  const attention = attentionParamsPerLayer(config)
  const denseFfnParamsPerLayer = intermediateSize > 0 ? 3 * hiddenSize * intermediateSize : 0

  // SwiGLU experts carry three projections: gate, up, and down.
  const paramsPerExpert = 3 * hiddenSize * moeIntermediateSize
  const paramsPerSharedExpert = 3 * hiddenSize * sharedExpertIntermediate
  const routerParamsPerLayer = hiddenSize * routedExperts

  const sharedExpertParamsPerLayer = sharedExperts * paramsPerSharedExpert
  const topK = expertsPerToken ?? 1

  const attentionParams = numLayers * attention
  const routedExpertParams = moeLayers * routedExperts * paramsPerExpert
  const sharedExpertParams = moeLayers * sharedExpertParamsPerLayer
  const denseFfnParams = denseLayers * denseFfnParamsPerLayer
  const routerParams = moeLayers * routerParamsPerLayer

  const embedParams = vocabSize * hiddenSize * (tiedEmbeddings ? 1 : 2)

  // The router is a single small matmul per layer, well under one percent of a
  // layer here. Leaving it out of the active figure keeps the effect of the
  // router top-k on compute visible, which is the question this tool answers.
  const activeParamsPerToken =
    attentionParams + moeLayers * (topK * paramsPerExpert + sharedExpertParamsPerLayer)

  const totalParams =
    attentionParams +
    routedExpertParams +
    sharedExpertParams +
    routerParams +
    denseFfnParams +
    embedParams

  return {
    modelType,
    hiddenSize,
    intermediateSize,
    moeIntermediateSize,
    routedExperts,
    expertsPerToken: topK,
    sharedExperts,
    sharedExpertIntermediate,
    numLayers,
    moeLayers,
    denseLayers,
    attentionParamsPerLayer: attention,
    denseFfnParamsPerLayer,
    routerParamsPerLayer,
    paramsPerExpert,
    paramsPerSharedExpert,
    activeParamsPerToken,
    totalParams,
    routedExpertParams,
    sharedExpertParams,
    attentionParams,
    denseFfnParams,
    embedParams,
    bestEffort,
    notes,
  }
}
