import { readBoolean, readNumber, readNumberFrom, readStringArray, unwrapConfig } from '../model-config'
import type { RawConfig } from '../model-config'

/**
 * The mixture of experts facts a config describes.
 *
 * Both calculators that size a model against a config need the same four things
 * out of it: how wide the experts are, how many there are, how many a token
 * routes to, and how many blocks hold a bank. Releases name all four
 * differently, so the reading is done once here and each tool composes its own
 * parameter count on top.
 */

/** Keys that name the routed expert count, in the order they are tried. */
export const ROUTED_EXPERT_KEYS = ['n_routed_experts', 'num_experts', 'num_local_experts'] as const

/** Keys that name the router top-k, in the order they are tried. */
export const EXPERTS_PER_TOKEN_KEYS = [
  'num_experts_per_tok',
  'num_experts_per_token',
  'experts_per_token',
] as const

export interface MoeShapeFields {
  /** The model_type of the language model, which nested configs keep inside. */
  modelType: string
  hiddenSize: number
  numLayers: number
  /** Dense feed forward width, used by the blocks that are not expert blocks. */
  intermediateSize: number
  /** Expert feed forward width. */
  moeIntermediateSize: number
  /** Routed experts per expert block. */
  routedExperts: number
  /** The router top-k, or undefined when the config names none. */
  expertsPerToken: number | undefined
  /** Always-on shared experts per expert block. */
  sharedExperts: number
  sharedExpertIntermediate: number
  vocabSize: number
  /** True when the config ties the embedding and the language model head. */
  tiedEmbeddings: boolean
  /** Blocks that hold an expert bank. */
  moeLayers: number
  /** Blocks whose feed forward is a plain dense MLP. */
  denseLayers: number
  /** Anything that had to be inferred rather than read, in the order found. */
  notes: string[]
}

/**
 * Reads the expert bank facts, or null when the config has no expert bank at
 * all. A dense model is a legitimate answer to "how many experts", not an
 * error, so the caller is told the difference.
 */
export function readMoeShapeFields(config: RawConfig): MoeShapeFields | null {
  const { inner, outer } = unwrapConfig(config)
  const notes: string[] = []

  const routedExperts = readNumberFrom(inner, ROUTED_EXPERT_KEYS) ?? 0
  if (routedExperts < 1) return null

  const hiddenSize = readNumber(inner, 'hidden_size') ?? 0
  if (hiddenSize < 1) return null

  const numLayers =
    readNumber(inner, 'num_hidden_layers') ?? readStringArray(inner, 'layers_block_type')?.length ?? 0
  if (numLayers < 1) return null

  const expertsPerToken = readNumberFrom(inner, EXPERTS_PER_TOKEN_KEYS)
  if (expertsPerToken === undefined) {
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

  return {
    modelType:
      typeof inner.model_type === 'string'
        ? inner.model_type
        : typeof outer.model_type === 'string'
          ? outer.model_type
          : 'unknown',
    hiddenSize,
    numLayers,
    intermediateSize,
    moeIntermediateSize,
    routedExperts,
    expertsPerToken,
    sharedExperts,
    sharedExpertIntermediate,
    vocabSize: readNumber(inner, 'vocab_size') ?? readNumber(outer, 'vocab_size') ?? 0,
    tiedEmbeddings: readBoolean(inner, 'tie_word_embeddings') === true,
    moeLayers,
    denseLayers: Math.max(0, numLayers - moeLayers),
    notes,
  }
}

/**
 * Counts the blocks that hold an expert bank.
 *
 * Configs describe this in four different ways, so each is tried in turn and
 * the reason is recorded when a fallback is used.
 */
export function countMoeLayers(
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
