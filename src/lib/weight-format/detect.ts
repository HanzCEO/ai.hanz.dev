import {
  readNumber,
  readNumberArray,
  readObject,
  readString,
  readStringArray,
  unwrapConfig,
  type RawConfig,
} from '../model-config'
import type { ModelShape } from '../model-shape'

import {
  DEFAULT_WEIGHT_FORMAT,
  bytesPerWeight,
  isFp8Format,
  weightFormatLabel,
} from './formats'
import type {
  WeightBytes,
  WeightFormatId,
  WeightQuantization,
  WeightQuantizationSource,
} from './types'

/** The FP8 block size a config omits. Most published FP8 checkpoints use 128 by 128. */
const DEFAULT_FP8_BLOCK_SIZE = 16384

function includesAny(value: string | undefined, needles: string[]): boolean {
  if (!value) return false
  const lower = value.toLowerCase()
  return needles.some((needle) => lower.includes(needle))
}

/**
 * Reads a format name the way a checkpoint writes it.
 *
 * A name can be a quant method, a storage dtype, or a torch dtype, so the
 * reader accepts all three vocabularies. FP4 without a variant is read as
 * MXFP4, which is the OCP format DeepSeek V4 and MiMo V2.6 store.
 */
function formatFromName(name: string | undefined): WeightFormatId | null {
  if (!name) return null
  if (includesAny(name, ['mxfp4'])) return 'MXFP4'
  if (includesAny(name, ['nvfp4'])) return 'NVFP4'
  if (includesAny(name, ['fp4', 'e2m1', '4bit'])) return 'MXFP4'
  if (includesAny(name, ['e5m2'])) return 'FP8_E5M2'
  if (includesAny(name, ['fp8', 'e4m3', 'float8'])) return 'FP8_E4M3'
  if (includesAny(name, ['int8'])) return 'INT8'
  if (includesAny(name, ['int4', 'gptq', 'awq'])) return 'INT4'
  if (includesAny(name, ['bfloat16', 'bf16'])) return 'BF16'
  if (includesAny(name, ['float16', 'fp16', 'half'])) return 'FP16'
  return null
}

/**
 * Reads the weight format a config describes.
 *
 * The reading is deliberately narrow about the fields it trusts. A quant
 * method of fp8 describes the arithmetic, not the stored weights, so a store
 * dtype wins over it. An expert dtype describes only the routed experts, so it
 * moves one bucket and leaves the other alone. A torch dtype is the compute
 * dtype, so it is the last resort rather than the first answer.
 *
 * The result always fills both buckets, which is what lets a caller cost a
 * DeepSeek V4 checkpoint as MXFP4 experts beside FP8 attention without a
 * branch for the mixed case.
 */
export function detectWeightQuantization(config: RawConfig): WeightQuantization {
  const { inner, outer } = unwrapConfig(config)

  const quant = readObject(outer, 'quantization_config') ?? readObject(inner, 'quantization_config')
  const storeDtype = quant ? readString(quant, 'store_dtype') : undefined
  const quantMethod = quant ? readString(quant, 'quant_method') : undefined
  const fmt = quant ? readString(quant, 'fmt') : undefined
  const expertDtype =
    readString(outer, 'expert_dtype') ??
    readString(inner, 'expert_dtype') ??
    (quant ? readString(quant, 'expert_dtype') : undefined)
  const dtypeName =
    readString(outer, 'torch_dtype') ??
    readString(inner, 'torch_dtype') ??
    readString(outer, 'dtype') ??
    readString(inner, 'dtype')

  const weightBlock = quant ? readNumberArray(quant, 'weight_block_size') : undefined
  const weightBlockSize =
    weightBlock && weightBlock.length > 0
      ? weightBlock.reduce((product, value) => product * value, 1)
      : undefined
  const fourBitBlockSize =
    (quant ? readNumber(quant, 'mxfp4_block_size') : undefined) ?? weightBlockSize

  // NVFP4 shares an E4M3 scale across 16 values, while MXFP4 shares an E8M0
  // scale across 32. A name alone is enough, and a name of fp4 beside an E4M3
  // scale on 16 values is the NVFP4 layout.
  const scaleFormat = (quant ? readString(quant, 'scale_fmt') : undefined) ?? fmt
  const nvfp4Named =
    includesAny(storeDtype, ['nvfp4']) || includesAny(quantMethod, ['nvfp4'])
  const nvfp4Context =
    nvfp4Named || (includesAny(scaleFormat, ['e4m3']) && fourBitBlockSize === 16)

  // The primary format is the one the checkpoint names for the whole model.
  const storeFormat = formatFromName(storeDtype)
  const methodFormat = formatFromName(fmt) ?? formatFromName(quantMethod)
  const dtypeFormat = formatFromName(dtypeName)

  // A store dtype names the stored weights, so it wins. A quant method names
  // the arithmetic, so it is next. A torch dtype is the compute dtype, so it
  // is the last resort before the BF16 default.
  let primary: WeightFormatId
  let source: WeightQuantizationSource
  if (storeFormat) {
    primary = storeFormat
    source = 'store_dtype'
  } else if (methodFormat) {
    primary = methodFormat
    source = 'quantization_config'
  } else if (dtypeFormat) {
    primary = dtypeFormat
    source = 'torch_dtype'
  } else {
    primary = DEFAULT_WEIGHT_FORMAT
    source = 'assumed'
  }

  // An expert dtype moves only the expert bucket. A four bit expert dtype is
  // MXFP4 unless the checkpoint elsewhere names the NVFP4 layout.
  let experts = primary
  if (expertDtype) {
    const lower = expertDtype.toLowerCase()
    if (lower.includes('fp4') || lower.includes('e2m1') || lower.includes('4bit')) {
      experts = nvfp4Context ? 'NVFP4' : 'MXFP4'
    } else {
      experts = formatFromName(expertDtype) ?? primary
    }
  }
  const dense = primary
  const mixed = experts !== dense

  const fp8BlockSize =
    isFp8Format(experts) || isFp8Format(dense)
      ? (weightBlockSize ?? DEFAULT_FP8_BLOCK_SIZE)
      : null

  const ignoredLayers =
    (quant ? readStringArray(quant, 'ignored_layers') : undefined) ??
    readStringArray(outer, 'ignored_layers') ??
    readStringArray(inner, 'ignored_layers')

  let note: string | null = null
  if (ignoredLayers && ignoredLayers.length > 0) {
    note = `The config keeps ${ignoredLayers.length.toLocaleString('en-US')} ${ignoredLayers.length === 1 ? 'layer' : 'layers'} in higher precision under ignored_layers. The calculator costs every weight in the detected format, so the footprint is a little understated.`
  } else if (!quant && !expertDtype) {
    note = `The config names no quantization, so the calculator costs the weights as ${weightFormatLabel(primary)}.`
  }

  return {
    primary,
    experts,
    dense,
    mixed,
    source,
    fp8BlockSize,
    note,
  }
}

/**
 * The quantization a reader forces from the picker.
 *
 * A forced format puts every bucket on one format, which is what a reader means
 * when they say the model is served in FP8. The automatic reading of the config
 * is what produces a mixed split.
 */
export function uniformWeightQuantization(id: WeightFormatId): WeightQuantization {
  return {
    primary: id,
    experts: id,
    dense: id,
    mixed: false,
    source: 'manual',
    fp8BlockSize: isFp8Format(id) ? DEFAULT_FP8_BLOCK_SIZE : null,
    note: null,
  }
}

/**
 * Bytes a shape needs in the detected formats, split by bucket.
 *
 * The expert bucket holds the routed and shared experts. The dense bucket
 * holds attention, the dense feed forward, the router, and the embeddings.
 * The active figure is what one token reads on the forward pass, with the
 * router excluded, which is the term the decode bandwidth roofline divides by.
 */
export function weightBytesFor(shape: ModelShape, quant: WeightQuantization): WeightBytes {
  const expertBytes = bytesPerWeight(quant.experts, { blockSize: quant.fp8BlockSize })
  const denseBytes = bytesPerWeight(quant.dense, { blockSize: quant.fp8BlockSize })

  const expertParams = shape.routedExpertParams + shape.sharedExpertParams
  const denseParams =
    shape.attentionParams +
    shape.denseFfnParams +
    shape.routerParams +
    shape.embedParams

  const experts = expertParams * expertBytes
  const dense = denseParams * denseBytes

  // One token reads its top-k experts in each expert block plus every shared
  // expert. The routed expert figure is the whole bank, so it is divided by
  // the bank size first.
  const activeExpertParams =
    shape.routedExperts > 0
      ? (shape.routedExpertParams / shape.routedExperts) * shape.expertsPerToken +
        shape.sharedExpertParams
      : shape.sharedExpertParams
  const activeDenseParams = shape.attentionParams + shape.denseFfnParams
  const activePerToken = activeDenseParams * denseBytes + activeExpertParams * expertBytes

  return {
    total: experts + dense,
    experts,
    dense,
    activePerToken,
    expertBytes,
    denseBytes,
  }
}
