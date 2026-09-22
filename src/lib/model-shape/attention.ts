import { readFlagArray, readNumber, readStringArray, unwrapConfig } from '../model-config'
import type { RawConfig } from '../model-config'

/**
 * Parameters in one attention block.
 *
 * Two layouts cover almost every release: grouped query attention, and the
 * compressed latent of multi head latent attention. Anything unrecognised falls
 * back to a plain transformer block of four hidden by hidden matrices.
 *
 * DeepSeek V4 style compressed attention has no kv_lora_rank, so it lands on the
 * grouped query path and its attention weights are overstated. Attention is a
 * small share of a large model, so the error stays inside the estimate.
 *
 * This is plain config arithmetic with no dependency on any particular pruning
 * or distillation method, which is why it lives outside the REAP estimator and
 * is shared by every calculator that has to size a block.
 */
export function attentionParamsPerLayer(config: RawConfig): number {
  const { inner, outer } = unwrapConfig(config)
  const hidden = readNumber(inner, 'hidden_size') ?? readNumber(outer, 'hidden_size') ?? 0
  if (hidden <= 0) return 0

  const numHeads = readNumber(inner, 'num_attention_heads') ?? 0
  const kvLoraRank = readNumber(inner, 'kv_lora_rank')

  if (kvLoraRank !== undefined && numHeads > 0) {
    return mlaParams(inner, hidden, numHeads, kvLoraRank)
  }

  return gqaParams(inner, hidden, '', '')
}

/**
 * Attention parameters across the whole model, not just one block.
 *
 * Most models repeat one attention block, so the count is the depth times the
 * block. Two cases break that. A model with a hybrid layer pattern runs two
 * geometries side by side, and a sliding window layer carries its own head
 * counts and head widths, so a single block cannot stand for all of them.
 * Getting this wrong on a 1T parameter model moves the total by billions of
 * parameters, which is why the total is its own function rather than a
 * multiplication at each call site.
 *
 * `hybrid_layer_pattern` marks each block: 0 is a global attention block, and 1
 * is a sliding window block. That is the convention MiMo-V2.6 uses.
 */
export function attentionParamsTotal(config: RawConfig): number {
  const { inner, outer } = unwrapConfig(config)
  const hidden = readNumber(inner, 'hidden_size') ?? readNumber(outer, 'hidden_size') ?? 0
  const numLayers =
    readNumber(inner, 'num_hidden_layers') ?? readStringArray(inner, 'layers_block_type')?.length ?? 0
  if (hidden <= 0 || numLayers < 1) return 0

  const numHeads = readNumber(inner, 'num_attention_heads') ?? 0
  const kvLoraRank = readNumber(inner, 'kv_lora_rank')

  if (kvLoraRank !== undefined && numHeads > 0) {
    return numLayers * mlaParams(inner, hidden, numHeads, kvLoraRank)
  }

  const pattern = readFlagArray(inner, 'hybrid_layer_pattern')
  if (pattern && pattern.length >= numLayers) {
    const slidingLayers = pattern.slice(0, numLayers).filter(Boolean).length
    const fullLayers = numLayers - slidingLayers
    return fullLayers * gqaParams(inner, hidden, '', '') + slidingLayers * gqaParams(inner, hidden, 'swa_', 'swa_')
  }

  return numLayers * attentionParamsPerLayer(config)
}

/**
 * One grouped query attention block.
 *
 * `headsPrefix` and `dimPrefix` name the field set to read. The sliding window
 * geometry of a hybrid is published under `swa_` names, and each of those falls
 * back to its base field so a config that names only some of them is still
 * sized. The value projection and the output projection use `v_head_dim`, which
 * is narrower than the query and key width on the models that set it.
 */
function gqaParams(
  inner: RawConfig,
  hidden: number,
  headsPrefix: string,
  dimPrefix: string,
): number {
  const numHeads =
    readNumber(inner, `${headsPrefix}num_attention_heads`) ??
    readNumber(inner, 'num_attention_heads') ??
    0
  const numKvHeads =
    readNumber(inner, `${headsPrefix}num_key_value_heads`) ??
    readNumber(inner, 'num_key_value_heads') ??
    numHeads
  const headDim =
    readNumber(inner, `${dimPrefix}head_dim`) ??
    readNumber(inner, 'head_dim') ??
    (numHeads > 0 ? Math.floor(hidden / numHeads) : 0)
  const vHeadDim =
    readNumber(inner, `${dimPrefix}v_head_dim`) ??
    readNumber(inner, 'v_head_dim') ??
    headDim

  if (numHeads > 0 && headDim > 0) {
    const vDim = vHeadDim > 0 ? vHeadDim : headDim
    // Query projection, key projection, value projection, output projection.
    return (
      hidden * numHeads * headDim +
      hidden * numKvHeads * headDim +
      hidden * numKvHeads * vDim +
      numHeads * vDim * hidden
    )
  }

  return 4 * hidden * hidden
}

/** Multi head latent attention, where the key and value pair is compressed. */
function mlaParams(
  inner: RawConfig,
  hidden: number,
  numHeads: number,
  kvLoraRank: number,
): number {
  const qkRope = readNumber(inner, 'qk_rope_head_dim') ?? 64
  const qkNope = readNumber(inner, 'qk_nope_head_dim') ?? 128
  const vHead = readNumber(inner, 'v_head_dim') ?? 128
  const qLoraRank = readNumber(inner, 'q_lora_rank')

  const qDim = numHeads * (qkNope + qkRope)
  const qProj = qLoraRank !== undefined ? hidden * qLoraRank + qLoraRank * qDim : hidden * qDim

  // Down projection into the latent, then up into per head key and value.
  const kvDown = hidden * (kvLoraRank + qkRope)
  const kvUp = kvLoraRank * (numHeads * (qkNope + vHead))
  const oProj = numHeads * vHead * hidden

  return qProj + kvDown + kvUp + oProj
}
