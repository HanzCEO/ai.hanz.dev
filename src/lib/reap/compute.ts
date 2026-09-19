import type { GpuSpec } from '../hardware'

import {
  RUNTIME_OVERHEAD_BYTES,
  WEIGHT_DTYPE_ORDER,
  bytesPerParam,
} from './presets'
import {
  ReapInputError,
  type MoeShape,
  type ReapBound,
  type ReapInputs,
  type ReapResult,
  type ReapVerdict,
  type WeightDtype,
} from './types'

const GIB = 1024 ** 3

/** Activations are held in BF16 regardless of the weight format. */
const ACTIVATION_BYTES = 2

/**
 * The activation buffer holds more than the residual stream: the gate and up
 * projections of the active experts, the input to the down projection, and the
 * router logits all live at once. Eight copies of the hidden state is a working
 * approximation of that peak.
 */
const ACTIVATION_FACTOR = 8

/** Above this the remaining expert bank stops resembling the original model. */
const MAX_PRUNE_RATIO = 0.9

/** The dense tensor rate the calibration forward pass actually runs at. */
function peakTflops(dtype: WeightDtype, gpu: GpuSpec): number {
  // A quantised checkpoint is dequantised per block, so it runs at the rate of
  // the nearest supported tensor format rather than at the BF16 rate.
  if ((dtype === 'FP8' || dtype === 'INT4') && gpu.fp8DenseTflops !== null) {
    return gpu.fp8DenseTflops
  }
  return gpu.bf16DenseTflops
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

function validate(inputs: ReapInputs): void {
  if (!Number.isInteger(inputs.calibrationSamples) || inputs.calibrationSamples < 1) {
    throw new ReapInputError(
      'Calibration samples must be a whole number of one or more.',
      'samples',
    )
  }
  if (!Number.isInteger(inputs.sequenceLength) || inputs.sequenceLength < 1) {
    throw new ReapInputError(
      'Sequence length must be a whole number of one or more.',
      'sequenceLength',
    )
  }
  if (
    !Number.isFinite(inputs.pruneRatio) ||
    inputs.pruneRatio < 0 ||
    inputs.pruneRatio > MAX_PRUNE_RATIO
  ) {
    throw new ReapInputError(
      `The pruning ratio must be between 0 and ${MAX_PRUNE_RATIO}. Past that the expert bank stops resembling the original model.`,
      'pruneRatio',
    )
  }
  if (!Number.isFinite(inputs.mfu) || inputs.mfu <= 0 || inputs.mfu > 1) {
    throw new ReapInputError('Model factory utilisation must be above 0 and at most 1.', 'mfu')
  }
  if (!Number.isFinite(inputs.overheadFactor) || inputs.overheadFactor < 1) {
    throw new ReapInputError('The overhead factor must be at least 1.', 'overheadFactor')
  }
  if (!Number.isInteger(inputs.microBatchSize) || inputs.microBatchSize < 1) {
    throw new ReapInputError(
      'The micro batch size must be a whole number of one or more.',
      'microBatchSize',
    )
  }
}

/** The result for a model that has no expert bank, so there is nothing to prune. */
function notMoeResult(inputs: ReapInputs): ReapResult {
  return {
    moe: false,
    verdict: 'not-moe',
    shape: null,
    tokens: 0,
    flops: 0,
    computeSeconds: 0,
    streamSeconds: 0,
    setupSeconds: 0,
    overheadFactor: inputs.overheadFactor,
    estimateSeconds: 0,
    bound: 'compute',
    weightBytes: 0,
    perMoELayerBytes: 0,
    activationBytes: 0,
    peakVramBytes: 0,
    vramBytes: inputs.gpu.vramGiB * GIB,
    narrowestFittingDtype: null,
    pruneRatio: inputs.pruneRatio,
    keptExperts: 0,
    removedExperts: 0,
    expertsPerTokenAfter: 0,
    totalParamsAfter: 0,
    expertParamsAfter: 0,
    reductionPercent: 0,
    activeParamsPerTokenAfter: 0,
    steps: [],
    constants: [],
    assumptions: [
      'REAP removes experts from a sparsely activated mixture of experts. This model has no expert bank, so there is nothing to prune.',
    ],
  }
}

/**
 * Estimates how long a REAP calibration run takes, whether it fits the card,
 * and how much smaller the pruned model gets.
 *
 * The dominant cost is the calibration forward pass, which streams the weights
 * through one decoder block at a time. The run is limited by whichever is
 * slower: the GPU doing the arithmetic, or storage feeding it the weights.
 */
export function estimateReap(shape: MoeShape | null, inputs: ReapInputs): ReapResult {
  validate(inputs)
  if (shape === null) return notMoeResult(inputs)

  const vramBytes = inputs.gpu.vramGiB * GIB

  // --- Cost of the calibration pass ---------------------------------------

  const tokens = inputs.calibrationSamples * inputs.sequenceLength
  const flops = 2 * tokens * shape.activeParamsPerToken

  const peak = peakTflops(inputs.weightDtype, inputs.gpu) * 1e12
  const computeSeconds = flops / (peak * inputs.mfu)

  const weightBytes = shape.totalParams * bytesPerParam(inputs.weightDtype)
  const streamSeconds = weightBytes / (inputs.storage.bandwidthGBs * 1e9)

  const bound: ReapBound = computeSeconds >= streamSeconds ? 'compute' : 'stream'
  const estimateSeconds =
    Math.max(computeSeconds, streamSeconds) * inputs.overheadFactor + inputs.setupSeconds

  // --- Peak memory, which is what decides whether the run is possible -----

  const perMoELayerParams =
    shape.routedExperts * shape.paramsPerExpert +
    shape.sharedExperts * shape.paramsPerSharedExpert +
    shape.attentionParamsPerLayer +
    shape.routerParamsPerLayer

  const perMoELayerBytes = perMoELayerParams * bytesPerParam(inputs.weightDtype)

  const tokensPerMicroBatch = inputs.microBatchSize * inputs.sequenceLength
  const activationBytes =
    tokensPerMicroBatch * shape.hiddenSize * ACTIVATION_BYTES * ACTIVATION_FACTOR

  const peakVramBytes = perMoELayerBytes + activationBytes + RUNTIME_OVERHEAD_BYTES

  const peakForDtype = (dtype: WeightDtype) =>
    perMoELayerParams * bytesPerParam(dtype) + activationBytes + RUNTIME_OVERHEAD_BYTES

  const narrowestFittingDtype =
    WEIGHT_DTYPE_ORDER.find((dtype) => peakForDtype(dtype) <= vramBytes) ?? null

  let verdict: ReapVerdict
  if (peakVramBytes <= vramBytes) verdict = 'fits'
  else if (peakForDtype('FP8') <= vramBytes) verdict = 'needs-fp8'
  else verdict = 'needs-offload'

  // --- What pruning leaves behind -----------------------------------------

  const ratio = inputs.pruneRatio
  const routedExperts = shape.routedExperts

  let keptExperts = clamp(Math.floor(routedExperts * (1 - ratio)), 1, routedExperts)
  if (!inputs.scaleTopK) {
    // A token cannot route to more experts than exist, so the kept count can
    // never fall below the router top-k.
    keptExperts = Math.max(keptExperts, Math.min(shape.expertsPerToken, routedExperts))
  }
  const removedExperts = routedExperts - keptExperts

  const expertsPerTokenAfter = inputs.scaleTopK
    ? clamp(Math.round(shape.expertsPerToken * (1 - ratio)), 1, keptExperts)
    : shape.expertsPerToken

  const expertParamsAfter = shape.moeLayers * keptExperts * shape.paramsPerExpert
  const routerParamsAfter = shape.moeLayers * shape.hiddenSize * keptExperts
  const routerParamsBefore = shape.moeLayers * shape.routerParamsPerLayer

  const totalParamsAfter =
    shape.totalParams -
    removedExperts * shape.moeLayers * shape.paramsPerExpert -
    (routerParamsBefore - routerParamsAfter)

  const reductionPercent =
    shape.totalParams > 0 ? ((shape.totalParams - totalParamsAfter) / shape.totalParams) * 100 : 0

  const activeParamsPerTokenAfter =
    shape.attentionParams +
    shape.moeLayers *
      (expertsPerTokenAfter * shape.paramsPerExpert +
        shape.sharedExperts * shape.paramsPerSharedExpert)

  // --- Explanations -------------------------------------------------------

  const steps = [
    {
      label: 'Calibration tokens',
      detail: `${inputs.calibrationSamples.toLocaleString('en-US')} samples times ${inputs.sequenceLength.toLocaleString('en-US')} tokens each is ${tokens.toLocaleString('en-US')} tokens. This product, not the sample count alone, is what drives the run.`,
    },
    {
      label: 'Active parameters per token',
      detail: `${shape.numLayers.toLocaleString('en-US')} blocks of attention, plus ${shape.moeLayers.toLocaleString('en-US')} expert banks each touching ${shape.expertsPerToken} of ${shape.routedExperts.toLocaleString('en-US')} experts, is ${shape.activeParamsPerToken.toLocaleString('en-US')} parameters per token.`,
    },
    {
      label: 'Arithmetic',
      detail: `2 times ${tokens.toLocaleString('en-US')} tokens times ${shape.activeParamsPerToken.toLocaleString('en-US')} parameters is ${flops.toExponential(3)} FLOPs.`,
    },
    {
      label: 'Compute time',
      detail: `${(peak / 1e12).toLocaleString('en-US')} TFLOPS dense at ${(inputs.mfu * 100).toFixed(0)} percent utilisation gives ${(computeSeconds / 3600).toFixed(2)} hours.`,
    },
    {
      label: 'Weight streaming time',
      detail: `${weightBytes.toLocaleString('en-US')} bytes of weights at ${inputs.storage.bandwidthGBs} GB/s is ${(streamSeconds / 3600).toFixed(2)} hours.`,
    },
    {
      label: 'Estimate',
      detail: `The slower of the two is the ${bound} bound, so ${(estimateSeconds / 3600).toFixed(2)} hours after the ${inputs.overheadFactor} times overhead and ${(inputs.setupSeconds / 60).toFixed(0)} minutes of setup.`,
    },
    {
      label: 'One block in memory',
      detail: `The largest resident set is a single decoder block: ${shape.routedExperts.toLocaleString('en-US')} experts plus attention and router is ${perMoELayerParams.toLocaleString('en-US')} parameters, or ${(perMoELayerBytes / GIB).toFixed(1)} GiB at ${inputs.weightDtype}.`,
    },
    {
      label: 'After pruning',
      detail: `Keeping ${keptExperts.toLocaleString('en-US')} of ${routedExperts.toLocaleString('en-US')} experts per layer removes ${(reductionPercent).toFixed(1)} percent of the parameters.`,
    },
  ]

  const constants = [
    { key: 'hidden_size', value: shape.hiddenSize, source: 'config' },
    { key: 'num_hidden_layers', value: shape.numLayers, source: 'config' },
    { key: 'moe_layers', value: shape.moeLayers, source: 'derived from the config layer pattern' },
    { key: 'routed_experts', value: shape.routedExperts, source: 'config' },
    { key: 'experts_per_token', value: shape.expertsPerToken, source: 'config' },
    { key: 'moe_intermediate_size', value: shape.moeIntermediateSize, source: 'config' },
    { key: 'shared_experts', value: shape.sharedExperts, source: 'config' },
    {
      key: 'params_per_expert',
      value: shape.paramsPerExpert,
      source: '3 x hidden_size x moe_intermediate_size, the gate, up, and down projections',
    },
    { key: 'attention_params_per_layer', value: shape.attentionParamsPerLayer, source: 'derived from the attention layout' },
    { key: 'total_params', value: shape.totalParams, source: 'sum over every layer, including embeddings' },
  ]

  const assumptions = [
    'Experts use SwiGLU, so each carries three projections: gate, up, and down. An expert with two projections would be a third smaller.',
    'Attention parameters are counted once and applied to every block. On a hybrid model whose linear layers differ from its full attention layers, this is an approximation.',
    'The router is left out of the active parameter count. It is a single small matmul per block, under one percent of the block here.',
    'Calibration uses the layer-wise observer, which keeps one decoder block resident at a time. That is what makes a single card enough.',
    'Model factory utilisation absorbs kernel efficiency. Dense tensor throughput is rarely reached in practice, so a third is a realistic target and the figure to trust is the estimate, not the raw FLOPs.',
    'The overhead factor covers the activation hooks, the saliency reduction, and the dataloader, none of which appear in the FLOPs figure.',
    'REAP shrinks the model in memory. It does not reduce the arithmetic per token unless the router top-k is reduced as well.',
    ...shape.notes,
  ]

  return {
    moe: true,
    verdict,
    shape,
    tokens,
    flops,
    computeSeconds,
    streamSeconds,
    setupSeconds: inputs.setupSeconds,
    overheadFactor: inputs.overheadFactor,
    estimateSeconds,
    bound,
    weightBytes,
    perMoELayerBytes,
    activationBytes,
    peakVramBytes,
    vramBytes,
    narrowestFittingDtype,
    pruneRatio: ratio,
    keptExperts,
    removedExperts,
    expertsPerTokenAfter,
    totalParamsAfter,
    expertParamsAfter,
    reductionPercent,
    activeParamsPerTokenAfter,
    steps,
    constants,
    assumptions,
  }
}
