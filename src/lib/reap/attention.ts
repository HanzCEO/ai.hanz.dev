import { readNumber, unwrapConfig } from '../model-config'
import type { RawConfig } from '../model-config'

/**
 * Parameters in one attention block.
 *
 * Two layouts cover almost every MoE release: grouped query attention, and the
 * compressed latent of multi head latent attention. Anything unrecognised falls
 * back to a plain transformer block of four hidden by hidden matrices.
 *
 * DeepSeek V4 style compressed attention has no kv_lora_rank, so it lands on the
 * grouped query path and its attention weights are overstated. Attention is a
 * small share of a large MoE, so the error stays inside the estimate.
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

  const numKvHeads = readNumber(inner, 'num_key_value_heads') ?? numHeads
  const headDim =
    readNumber(inner, 'head_dim') ?? (numHeads > 0 ? Math.floor(hidden / numHeads) : 0)

  if (numHeads > 0 && headDim > 0) {
    // Query projection, key and value projections, output projection.
    return hidden * numHeads * headDim + 2 * hidden * numKvHeads * headDim + numHeads * headDim * hidden
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
