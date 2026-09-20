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
      'Calibration samples must be a whole number of 1 or more.',
      'samples',
    )
  }
  if (!Number.isInteger(inputs.sequenceLength) || inputs.sequenceLength < 1) {
    throw new ReapInputError(
      'Sequence length must be a whole number of 1 or more.',
      'sequenceLength',
    )
  }
  if (
    !Number.isFinite(inputs.pruneRatio) ||
    inputs.pruneRatio < 0 ||
    inputs.pruneRatio > MAX_PRUNE_RATIO
  ) {
    throw new ReapInputError(
      `The pruning ratio must be between 0 and ${MAX_PRUNE_RATIO}. A higher value removes too much of the expert bank.`,
      'pruneRatio',
    )
  }
  if (!Number.isFinite(inputs.mfu) || inputs.mfu <= 0 || inputs.mfu > 1) {
    throw new ReapInputError('Model factory utilisation must be more than 0 and at most 1.', 'mfu')
  }
  if (!Number.isFinite(inputs.overheadFactor) || inputs.overheadFactor < 1) {
    throw new ReapInputError('The overhead factor must be at least 1.', 'overheadFactor')
  }
  if (!Number.isInteger(inputs.microBatchSize) || inputs.microBatchSize < 1) {
    throw new ReapInputError(
      'The micro batch size must be a whole number of 1 or more.',
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
      'REAP removes experts from a sparsely activated mixture of experts. This model has no expert bank. Therefore the calculator has nothing to prune.',
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
      detail: `${inputs.calibrationSamples.toLocaleString('en-US')} samples times ${inputs.sequenceLength.toLocaleString('en-US')} tokens each gives ${tokens.toLocaleString('en-US')} tokens. This product, and not the sample count alone, sets the cost of the run.`,
    },
    {
      label: 'Active parameters per token',
      detail: `The model has ${shape.numLayers.toLocaleString('en-US')} blocks, and each one has attention. The ${shape.moeLayers.toLocaleString('en-US')} expert banks each send a token to ${shape.expertsPerToken} of ${shape.routedExperts.toLocaleString('en-US')} experts. One token therefore touches ${shape.activeParamsPerToken.toLocaleString('en-US')} parameters.`,
    },
    {
      label: 'Arithmetic',
      detail: `2 times ${tokens.toLocaleString('en-US')} tokens times ${shape.activeParamsPerToken.toLocaleString('en-US')} parameters gives ${flops.toExponential(3)} FLOPs.`,
    },
    {
      label: 'Arithmetic duration',
      detail: `The GPU reaches ${(peak / 1e12).toLocaleString('en-US')} TFLOPS dense. At ${(inputs.mfu * 100).toFixed(0)} percent utilisation, the arithmetic takes ${(computeSeconds / 3600).toFixed(2)} hours.`,
    },
    {
      label: 'Weight streaming duration',
      detail: `The run reads ${weightBytes.toLocaleString('en-US')} bytes of weights from storage at ${inputs.storage.bandwidthGBs} GB/s. That takes ${(streamSeconds / 3600).toFixed(2)} hours.`,
    },
    {
      label: 'Estimate',
      detail: `The slower value sets the bound, and that bound is ${bound === 'compute' ? 'the arithmetic' : 'the weight stream'}. The estimate is ${(estimateSeconds / 3600).toFixed(2)} hours, after ${inputs.overheadFactor} times overhead and ${(inputs.setupSeconds / 60).toFixed(0)} minutes of setup.`,
    },
    {
      label: 'One expert block in VRAM',
      detail: `The largest resident set is 1 expert block. ${shape.routedExperts.toLocaleString('en-US')} experts plus attention and the router gives ${perMoELayerParams.toLocaleString('en-US')} parameters. That is ${(perMoELayerBytes / GIB).toFixed(1)} GiB at ${inputs.weightDtype}.`,
    },
    {
      label: 'After pruning',
      detail: `The run keeps ${keptExperts.toLocaleString('en-US')} of ${routedExperts.toLocaleString('en-US')} experts in each expert block. That removes ${(reductionPercent).toFixed(1)} percent of the parameters.`,
    },
  ]

  const constants = [
    { key: 'hidden_size', value: shape.hiddenSize, source: 'config' },
    { key: 'num_hidden_layers', value: shape.numLayers, source: 'config' },
    { key: 'moe_layers', value: shape.moeLayers, source: 'derived from the layer pattern in the config' },
    { key: 'routed_experts', value: shape.routedExperts, source: 'config' },
    { key: 'experts_per_token', value: shape.expertsPerToken, source: 'config' },
    { key: 'moe_intermediate_size', value: shape.moeIntermediateSize, source: 'config' },
    { key: 'shared_experts', value: shape.sharedExperts, source: 'config' },
    {
      key: 'params_per_expert',
      value: shape.paramsPerExpert,
      source: '3 x hidden_size x moe_intermediate_size. These are the gate, up, and down projections.',
    },
    { key: 'attention_params_per_layer', value: shape.attentionParamsPerLayer, source: 'derived from the attention layout' },
    { key: 'total_params', value: shape.totalParams, source: 'the sum over every block, including the embeddings' },
  ]

  const assumptions = [
    'The experts use SwiGLU. Each expert therefore carries 3 projections: gate, up, and down. An expert with 2 projections would be one third smaller.',
    'The calculator counts the attention parameters once and applies them to every block. A hybrid model has linear layers that differ from its full attention layers. For that model, this is an approximation.',
    'The calculator leaves the router out of the active parameter count. The router is 1 small matmul in each block. That is less than 1 percent of the block here.',
    'The calibration uses the layer-wise observer. That observer keeps one expert block in VRAM at a time. Therefore one GPU is sufficient.',
    'Model factory utilisation covers the kernel efficiency. Real runs rarely reach the dense throughput, so one third is a realistic value. Trust the estimate and not the raw FLOPs.',
    'The overhead factor covers the activation hooks, the saliency reduction, and the dataloader. None of these appear in the FLOPs.',
    'REAP makes the model smaller in VRAM. It does not reduce the arithmetic for each token unless the run also reduces the router top-k.',
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
