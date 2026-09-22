/**
 * The weight formats a checkpoint can be published in.
 *
 * Every calculator that sizes weights against VRAM or memory bandwidth reads
 * one of these ids, so the byte cost of a format is defined once and shared
 * rather than repeated as a constant in each estimator. A format carries its
 * scale sidecar in the byte figure, because a 4 bit value is never stored
 * alone: it shares a scale with a block of neighbours.
 */
export type WeightFormatId =
  | 'BF16'
  | 'FP16'
  | 'FP8_E4M3'
  | 'FP8_E5M2'
  | 'INT8'
  | 'MXFP4'
  | 'NVFP4'
  | 'INT4'

/**
 * One weight format, with the byte figure that the size formulas consume.
 *
 * `bytes` includes the scale sidecar at the format's canonical block size.
 * `bytesPerWeight` recomputes that figure when a config names a different
 * block, which is what an FP8 checkpoint does.
 */
export interface WeightFormatSpec {
  id: WeightFormatId
  label: string
  /** Bytes for each weight, the payload and its share of the scale. */
  bytes: number
  /** The block size the byte figure assumes, or null when the format has no scale. */
  blockSize: number | null
  note: string
}

/** Where the detected format came from, so the breakdown can name the field. */
export type WeightQuantizationSource =
  | 'store_dtype'
  | 'expert_dtype'
  | 'quantization_config'
  | 'torch_dtype'
  | 'manual'
  | 'assumed'

/**
 * The format a config describes, split into the two parameter buckets that a
 * mixed checkpoint needs.
 *
 * A DeepSeek V4 checkpoint puts its routed experts on MXFP4 and everything
 * else on FP8. A MiMo V2.6 checkpoint stores most weights as MXFP4. A plain
 * BF16 checkpoint puts both buckets on BF16. The two buckets are always
 * present, so a caller never has to branch on whether the model is mixed.
 */
export interface WeightQuantization {
  /** The format named for the whole model, before the expert split. */
  primary: WeightFormatId
  /** The format for the routed and shared experts. */
  experts: WeightFormatId
  /** The format for attention, the dense feed forward, the router, and the embeddings. */
  dense: WeightFormatId
  /** True when the expert bucket and the dense bucket differ. */
  mixed: boolean
  /** The field the format was read from. */
  source: WeightQuantizationSource
  /** The FP8 block size the scale overhead uses, or null when the dense format has no block. */
  fp8BlockSize: number | null
  /** A note when the detection was partial, or null. */
  note: string | null
}

/** The bytes a shape needs in one weight format, split by bucket. */
export interface WeightBytes {
  /** Bytes for every weight in the checkpoint. */
  total: number
  /** Bytes held by the routed and shared experts. */
  experts: number
  /** Bytes held by attention, the dense feed forward, the router, and the embeddings. */
  dense: number
  /** Bytes one token reads on the forward pass, for the decode bandwidth roofline. */
  activePerToken: number
  /** Bytes for each expert weight, including its scale share. */
  expertBytes: number
  /** Bytes for each dense weight, including its scale share. */
  denseBytes: number
}
