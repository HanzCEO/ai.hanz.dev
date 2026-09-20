import { readBoolean, readNumber, unwrapConfig } from '../model-config'
import type { RawConfig } from '../model-config'

import { attentionParamsPerLayer } from './attention'
import { readMoeShapeFields } from './moe'

/**
 * The full shape of a model, in the terms a serving process needs.
 *
 * The other detectors answer narrower questions. The REAP detector counts the
 * experts that a pruning run removes, and the DSpark detector counts the target
 * that a drafter attaches to. This one counts the complete set of weights a
 * serving process has to keep resident, which is why it adds the shared
 * experts, the router, and the language model head separately from the active
 * path.
 *
 * Everything here is derivable from a config, so a hardware estimate needs no
 * checkpoint download and no measurement.
 */
export interface ModelShape {
  modelType: string
  hiddenSize: number
  numLayers: number
  /** Dense feed forward width, used by the blocks that are not expert blocks. */
  intermediateSize: number
  vocabSize: number
  /** True when the config ties the embedding and the language model head. */
  tiedEmbeddings: boolean
  numAttentionHeads: number
  numKeyValueHeads: number
  headDim: number
  /** Parameters in one attention block. */
  attentionParamsPerLayer: number

  /** Blocks whose feed forward is a plain dense MLP. */
  denseLayers: number
  /** Blocks that hold an expert bank. */
  moeLayers: number
  /** Routed experts per expert block. Zero on a dense model. */
  routedExperts: number
  /** Experts one token routes to. */
  expertsPerToken: number
  /** Expert feed forward width. */
  moeIntermediateSize: number
  /** Always-on shared experts per expert block. */
  sharedExperts: number
  sharedExpertIntermediate: number

  attentionParams: number
  denseFfnParams: number
  routedExpertParams: number
  sharedExpertParams: number
  routerParams: number
  embedParams: number

  /** Every weight in the checkpoint. */
  totalParams: number
  /** The weights one token reads on the forward pass, router excluded. */
  activeParamsPerToken: number

  /** True when a field had to be inferred rather than read. */
  bestEffort: boolean
  notes: string[]
}

/**
 * Reads the shape a config describes, or returns null when it describes no
 * transformer.
 *
 * A config with no hidden size or no depth is not a language model, so there is
 * nothing to size and the caller is told so rather than handed a zero.
 */
export function detectModelShape(config: RawConfig): ModelShape | null {
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

  const declaredIntermediate = readNumber(inner, 'intermediate_size')
  const intermediateSize = declaredIntermediate ?? 4 * hiddenSize
  if (declaredIntermediate === undefined) {
    notes.push(
      `The config has no intermediate_size. The dense feed forward width is therefore taken as four times the hidden width, which is ${intermediateSize.toLocaleString('en-US')}.`,
    )
  }

  const declaredVocab = readNumber(inner, 'vocab_size') ?? readNumber(outer, 'vocab_size')
  const vocabSize = declaredVocab ?? 0
  if (declaredVocab === undefined) {
    bestEffort = true
    notes.push(
      'The config has no vocab_size, so the embedding and the language model head are understated. Both scale with the vocabulary.',
    )
  }

  const declaredTied = readBoolean(inner, 'tie_word_embeddings')
  const tiedEmbeddings = declaredTied === true
  if (declaredTied === undefined) {
    bestEffort = true
    notes.push(
      'The config has no tie_word_embeddings, so the calculator counts the embedding and the language model head separately. A tied model is one embedding table smaller.',
    )
  }

  const numAttentionHeads = readNumber(inner, 'num_attention_heads') ?? 0
  const numKeyValueHeads = readNumber(inner, 'num_key_value_heads') ?? numAttentionHeads
  const headDim =
    readNumber(inner, 'head_dim') ??
    (numAttentionHeads > 0 ? Math.floor(hiddenSize / numAttentionHeads) : 0)

  const attention = attentionParamsPerLayer(config)

  // --- The expert bank, when the model has one ----------------------------
  //
  // The shared reader returns null for a dense model, which is a legitimate
  // answer rather than a failure. Every expert term is then zero.

  const moe = readMoeShapeFields(config)
  const moeLayers = moe?.moeLayers ?? 0
  const denseLayers = numLayers - moeLayers
  const routedExperts = moe?.routedExperts ?? 0
  const expertsPerToken = moe?.expertsPerToken ?? 0
  const moeIntermediateSize = moe?.moeIntermediateSize ?? intermediateSize
  const sharedExperts = moe?.sharedExperts ?? 0
  const sharedExpertIntermediate = moe?.sharedExpertIntermediate ?? moeIntermediateSize
  if (moe) notes.push(...moe.notes)

  // SwiGLU experts carry three projections: gate, up, and down.
  const paramsPerExpert = 3 * hiddenSize * moeIntermediateSize
  const paramsPerSharedExpert = 3 * hiddenSize * sharedExpertIntermediate

  const attentionParams = numLayers * attention
  const denseFfnParams = denseLayers * 3 * hiddenSize * intermediateSize
  const routedExpertParams = moeLayers * routedExperts * paramsPerExpert
  const sharedExpertParams = moeLayers * sharedExperts * paramsPerSharedExpert
  const routerParams = moeLayers * hiddenSize * routedExperts

  // An untied model keeps a separate language model head, so the vocabulary is
  // stored twice. Both tables have to be resident at serving time, so both are
  // counted.
  const embedParams = vocabSize * hiddenSize * (tiedEmbeddings ? 1 : 2)

  const totalParams =
    attentionParams +
    denseFfnParams +
    routedExpertParams +
    sharedExpertParams +
    routerParams +
    embedParams

  // The router is a single small matmul per block, well under one percent of a
  // block here. Leaving it out of the active figure keeps the effect of the
  // router top-k on throughput visible.
  const activeParamsPerToken =
    attentionParams +
    denseFfnParams +
    moeLayers * (expertsPerToken * paramsPerExpert + sharedExperts * paramsPerSharedExpert)

  return {
    modelType,
    hiddenSize,
    numLayers,
    intermediateSize,
    vocabSize,
    tiedEmbeddings,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    attentionParamsPerLayer: attention,
    denseLayers,
    moeLayers,
    routedExperts,
    expertsPerToken,
    moeIntermediateSize,
    sharedExperts,
    sharedExpertIntermediate,
    attentionParams,
    denseFfnParams,
    routedExpertParams,
    sharedExpertParams,
    routerParams,
    embedParams,
    totalParams,
    activeParamsPerToken,
    bestEffort,
    notes,
  }
}
