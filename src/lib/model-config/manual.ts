import type { RawConfig } from './types'

/**
 * The manual entry path, which describes a model as a config object.
 *
 * A reader with a private or unreleased model has no config to fetch. Rather
 * than a second set of formulas, the values they type are written into a config
 * and read back by the same detector every other mode uses. All three model
 * sources therefore agree by construction.
 */

/** The fields a reader can type to describe a model shape. */
export interface ManualShapeInputs {
  hiddenSize: string
  intermediateSize: string
  numLayers: string
  vocabSize: string
  attentionHeads: string
  kvHeads: string
  headDim: string
  routedExperts: string
  expertsPerToken: string
  moeIntermediateSize: string
  moeLayers: string
  /**
   * Whether the embedding and the language model head share one table.
   *
   * This value moves the weight figure by one embedding table, which is more
   * than a gibibyte on an 8B model. A config carries it, so manual entry has to
   * be able to state it as well.
   */
  tieEmbeddings: 'tied' | 'untied'
}

/**
 * Qwen3-8B, so manual entry opens on a real model rather than on zeros.
 *
 * hidden 4096, 36 blocks, a dense width of 12288, a vocabulary of 151936, 32
 * attention heads and 8 key and value heads at a head width of 128, untied.
 */
export const MANUAL_DEFAULTS: ManualShapeInputs = {
  hiddenSize: '4096',
  intermediateSize: '12288',
  numLayers: '36',
  vocabSize: '151936',
  attentionHeads: '32',
  kvHeads: '8',
  headDim: '128',
  routedExperts: '0',
  expertsPerToken: '0',
  moeIntermediateSize: '0',
  moeLayers: '0',
  tieEmbeddings: 'untied',
}

function numberFrom(value: string): number | null {
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

function optionalFrom(value: string): number | undefined {
  return numberFrom(value) ?? undefined
}

/**
 * Turns the manual fields into a config shape.
 *
 * An expert bank is only described when a routed expert count is given, which
 * is what keeps a dense model off the mixture of experts path.
 */
export function manualConfig(inputs: ManualShapeInputs): RawConfig | null {
  const hiddenSize = numberFrom(inputs.hiddenSize)
  const numLayers = numberFrom(inputs.numLayers)
  if (!hiddenSize || !numLayers) return null

  const config: RawConfig = {
    model_type: 'manual',
    hidden_size: hiddenSize,
    num_hidden_layers: numLayers,
    tie_word_embeddings: inputs.tieEmbeddings === 'tied',
  }

  const intermediateSize = optionalFrom(inputs.intermediateSize)
  if (intermediateSize) config.intermediate_size = intermediateSize
  const vocabSize = optionalFrom(inputs.vocabSize)
  if (vocabSize) config.vocab_size = vocabSize
  const attentionHeads = optionalFrom(inputs.attentionHeads)
  if (attentionHeads) config.num_attention_heads = attentionHeads
  const headDim = optionalFrom(inputs.headDim)
  if (headDim) config.head_dim = headDim
  const kvHeads = optionalFrom(inputs.kvHeads)
  if (kvHeads) config.num_key_value_heads = kvHeads

  const routedExperts = numberFrom(inputs.routedExperts)
  if (routedExperts) {
    config.n_routed_experts = routedExperts
    const expertsPerToken = optionalFrom(inputs.expertsPerToken)
    if (expertsPerToken) config.num_experts_per_tok = expertsPerToken
    const expertWidth = optionalFrom(inputs.moeIntermediateSize)
    if (expertWidth) config.moe_intermediate_size = expertWidth
    const moeLayers = numberFrom(inputs.moeLayers)
    if (moeLayers) {
      // The detector reads the expert block count as the layers that are not
      // dense, so the dense count is what has to be recorded here.
      config.first_k_dense_replace = Math.max(0, numLayers - moeLayers)
    }
  }

  return config
}
