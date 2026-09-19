import {
  readBoolean,
  readNumber,
  readNumberFrom,
  readStringArray,
  unwrapConfig,
} from '../model-config'
import type { RawConfig } from '../model-config'

import { attentionParamsPerLayer } from './attention'
import type { MoeShape } from './types'

/** Keys that name the routed expert count, in the order they are tried. */
const ROUTED_EXPERT_KEYS = ['n_routed_experts', 'num_experts', 'num_local_experts'] as const

/** Keys that name the router top-k, in the order they are tried. */
const EXPERTS_PER_TOKEN_KEYS = [
  'num_experts_per_tok',
  'num_experts_per_token',
  'experts_per_token',
] as const

/**
 * Reads the MoE shape out of a config, or returns null when the model has no
 * expert bank.
 *
 * REAP only applies to a sparsely activated mixture of experts. A dense model
 * has nothing to prune, so the caller is told so rather than being handed a
 * meaningless number.
 */
export function detectMoeShape(config: RawConfig): MoeShape | null {
  const { inner, outer } = unwrapConfig(config)
  const notes: string[] = []
  let bestEffort = false

  const routedExperts = readNumberFrom(inner, ROUTED_EXPERT_KEYS) ?? 0
  if (routedExperts < 1) return null

  const hiddenSize = readNumber(inner, 'hidden_size') ?? 0
  if (hiddenSize < 1) return null

  const numLayers =
    readNumber(inner, 'num_hidden_layers') ??
    readStringArray(inner, 'layers_block_type')?.length ??
    0
  if (numLayers < 1) return null

  const expertsPerToken = readNumberFrom(inner, EXPERTS_PER_TOKEN_KEYS)
  if (expertsPerToken === undefined) {
    bestEffort = true
    notes.push(
      'The config does not name a router top-k, so one expert per token is assumed. Active parameters are understated.',
    )
  }

  const intermediateSize = readNumber(inner, 'intermediate_size') ?? 0
  const moeIntermediateSize = readNumber(inner, 'moe_intermediate_size') ?? intermediateSize
  if (readNumber(inner, 'moe_intermediate_size') === undefined) {
    notes.push(
      `The config has no moe_intermediate_size, so the dense width of ${intermediateSize.toLocaleString('en-US')} is used for each expert.`,
    )
  }

  const sharedExpertIntermediate =
    readNumber(inner, 'shared_expert_intermediate_size') ?? moeIntermediateSize
  const sharedExperts =
    readNumber(inner, 'n_shared_experts') ??
    (readNumber(inner, 'shared_expert_intermediate_size') !== undefined ? 1 : 0)

  const moeLayers = countMoeLayers(inner, numLayers, routedExperts, notes)
  const denseLayers = Math.max(0, numLayers - moeLayers)

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

  const vocabSize = readNumber(inner, 'vocab_size') ?? readNumber(outer, 'vocab_size') ?? 0
  const tied = readBoolean(inner, 'tie_word_embeddings') === true
  const embedParams = vocabSize * hiddenSize * (tied ? 1 : 2)

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
    modelType:
      typeof inner.model_type === 'string'
        ? inner.model_type
        : typeof outer.model_type === 'string'
          ? outer.model_type
          : 'unknown',
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

/**
 * Counts the blocks that hold an expert bank.
 *
 * Configs describe this in four different ways, so each is tried in turn and
 * the reason is recorded when a fallback is used.
 */
function countMoeLayers(
  inner: RawConfig,
  numLayers: number,
  routedExperts: number,
  notes: string[],
): number {
  if (routedExperts < 1) return 0

  // A per layer list is the most explicit form.
  const layerTypes = readStringArray(inner, 'mlp_layer_types')
  if (layerTypes) {
    return layerTypes.filter((type) => type === 'sparse' || type === 'moe').length
  }

  // Layers that are explicitly dense are subtracted from the total.
  const denseOnly = readStringArray(inner, 'mlp_only_layers')?.length ?? 0

  const firstKDense = readNumber(inner, 'first_k_dense_replace')
  if (firstKDense !== undefined) {
    return Math.max(0, numLayers - firstKDense - denseOnly)
  }

  const freq = readNumber(inner, 'moe_layer_freq')
  if (freq !== undefined && freq > 1) {
    notes.push(`moe_layer_freq is ${freq}, so every ${freq}th block holds an expert bank.`)
    return Math.max(0, Math.floor(numLayers / freq) - denseOnly)
  }

  return Math.max(0, numLayers - denseOnly)
}
