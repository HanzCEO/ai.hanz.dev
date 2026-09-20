import { formatBytes, formatExact } from '../format'
import {
  ACTIVATION_BYTES_PER_ELEMENT,
  ACTIVATION_FACTOR,
  BF16_BYTES,
  GRADIENT_BYTES_PER_PARAM,
  INT32_BYTES,
  MAX_SUGGESTED_GPUS,
  OPTIMIZER_BYTES_PER_PARAM,
  RUNTIME_OVERHEAD_BYTES,
  UINT8_BYTES,
} from './presets'
import {
  DsparkInputError,
  type DsparkBound,
  type DsparkInputs,
  type DsparkInputField,
  type DsparkResult,
  type DsparkTargetShape,
  type DsparkVerdict,
} from './types'

const GIB = 1024 ** 3

/** The scale a value has to reach to be a cost worth reporting. */
function requirePositive(value: number, field: DsparkInputField, label: string): void {
  if (!Number.isFinite(value) || value < 1) {
    throw new DsparkInputError(`${label} must be a whole number of one or more.`, field)
  }
}

function validate(inputs: DsparkInputs): void {
  requirePositive(inputs.trainingTokens, 'trainingTokens', 'Training tokens')
  requirePositive(inputs.epochs, 'epochs', 'Epochs')
  requirePositive(inputs.numAnchors, 'numAnchors', 'Anchors per sequence')
  requirePositive(inputs.blockSize, 'blockSize', 'Block size')
  requirePositive(inputs.numDraftLayers, 'numDraftLayers', 'Draft layers')
  requirePositive(inputs.numTargetLayers, 'numTargetLayers', 'Captured target layers')
  requirePositive(inputs.sequenceLength, 'sequenceLength', 'Sequence length')
  requirePositive(inputs.gpuCount, 'gpuCount', 'GPUs')
  requirePositive(inputs.microBatchSize, 'microBatchSize', 'Micro batch size')

  if (!Number.isFinite(inputs.markovRank) || inputs.markovRank < 0) {
    throw new DsparkInputError(
      'Markov rank must be zero or more. Zero disables the sequential head.',
      'markovRank',
    )
  }
  if (!Number.isFinite(inputs.mfu) || inputs.mfu <= 0 || inputs.mfu > 1) {
    throw new DsparkInputError(
      'Model flops utilisation must be above zero and at most one.',
      'mfu',
    )
  }
  if (!Number.isFinite(inputs.overheadFactor) || inputs.overheadFactor < 1) {
    throw new DsparkInputError('Overhead must be at least one.', 'overheadFactor')
  }
}

/**
 * Estimates what it costs to train a DSpark drafter against a target model.
 *
 * The three questions are how much storage the target cache takes, how long the
 * run lasts, and whether it fits the card. They are answered from the target's
 * config plus the training recipe, with no measurement required.
 *
 * Two things dominate and they pull in opposite directions. Offline training
 * precomputes the target's hidden states, which is where the tens of terabytes
 * come from, and then reads them back once per epoch, which is where the time
 * goes on a slow disk. Online training skips the storage entirely but has to
 * keep the whole target resident for the duration, which is where the VRAM goes.
 * The verdict says which of those two you can afford.
 */
export function estimateDspark(shape: DsparkTargetShape, inputs: DsparkInputs): DsparkResult {
  validate(inputs)

  const offline = inputs.dataMode === 'offline'
  const vramBytes = inputs.gpu.vramGiB * GIB
  const { hiddenSize, vocabSize } = shape

  // --- The target cache ---------------------------------------------------
  //
  // One row per token, matching the layout DeepSpec writes: the captured hidden
  // states from the chosen layers in bf16, the last hidden state in bf16, which
  // the distribution and confidence losses both need, then the token ids in
  // int32 and the two masks in uint8.
  const cacheBytesPerToken =
    inputs.numTargetLayers * hiddenSize * BF16_BYTES +
    hiddenSize * BF16_BYTES +
    INT32_BYTES +
    UINT8_BYTES +
    UINT8_BYTES

  const cacheBytes = offline ? cacheBytesPerToken * inputs.trainingTokens : 0
  const storageBytesPerSecond = inputs.storage.bandwidthGBs * 1e9
  const cacheWriteSeconds = cacheBytes / storageBytesPerSecond

  // --- The draft model ----------------------------------------------------
  //
  // The backbone is a shallow transformer over the block. It shares and freezes
  // the target's embedding and language model head, so neither appears in the
  // trained parameter count, but the projection that fuses the captured target
  // layers into the draft width does, and so do the two heads.
  const draftBackboneParams =
    inputs.numDraftLayers *
    (shape.attentionParamsPerLayer + 3 * hiddenSize * shape.intermediateSize)
  const draftProjectionParams = inputs.numTargetLayers * hiddenSize * hiddenSize
  const draftMarkovParams = 2 * vocabSize * inputs.markovRank
  const draftConfidenceParams = hiddenSize + inputs.markovRank
  const draftParams =
    draftBackboneParams + draftProjectionParams + draftMarkovParams + draftConfidenceParams

  const draftWeightBytes = draftParams * BF16_BYTES
  const optimizerBytes = draftParams * OPTIMIZER_BYTES_PER_PARAM
  const gradientBytes = draftParams * GRADIENT_BYTES_PER_PARAM

  // --- Arithmetic ---------------------------------------------------------
  //
  // Training samples anchor blocks rather than running over the sequence, so
  // the positions actually scored are the sequence count times the anchors per
  // sequence times the tokens per block. That is far fewer positions than there
  // are tokens, which is what makes a drafter cheap to train despite the cache
  // being enormous.
  const sequencesPerEpoch = inputs.trainingTokens / inputs.sequenceLength
  const positionsPerEpoch = sequencesPerEpoch * inputs.numAnchors * inputs.blockSize

  // Six flops per parameter per position: a forward and a backward pass.
  const trainingFlops = 6 * positionsPerEpoch * inputs.epochs * draftParams

  // Each block attends to the target context before its anchor, which is a
  // quadratic term in the context length and does not shrink with the anchors.
  const contextFlops =
    4 *
    inputs.numAnchors *
    inputs.blockSize *
    inputs.sequenceLength *
    hiddenSize *
    inputs.numDraftLayers *
    sequencesPerEpoch *
    inputs.epochs

  // Building the cache is one forward pass of the target over every token.
  const cachePrepFlops = offline ? 2 * inputs.trainingTokens * shape.activeParamsPerToken : 0

  const totalFlops = trainingFlops + contextFlops + cachePrepFlops

  const peak = inputs.gpu.bf16DenseTflops * 1e12 * inputs.gpuCount
  const computeSeconds = totalFlops / (peak * inputs.mfu)

  // The cache is read back once per epoch, so the stream cost scales with the
  // epoch count while the write happens only once.
  const cacheReadSeconds = (cacheBytes * inputs.epochs) / storageBytesPerSecond

  const bound: DsparkBound = computeSeconds >= cacheReadSeconds ? 'compute' : 'cache-read'
  const estimateSeconds =
    Math.max(computeSeconds, cacheReadSeconds) * inputs.overheadFactor + inputs.setupSeconds

  // --- Memory -------------------------------------------------------------
  const activationBytes =
    inputs.microBatchSize *
    inputs.numAnchors *
    inputs.blockSize *
    hiddenSize *
    ACTIVATION_BYTES_PER_ELEMENT *
    ACTIVATION_FACTOR

  // Online capture means the target stays loaded for the whole run. Offline
  // means it ran once, during cache preparation, and is gone by training time.
  const targetWeightBytes = offline ? 0 : shape.totalParams * BF16_BYTES

  const peakVramBytes =
    draftWeightBytes +
    optimizerBytes +
    gradientBytes +
    activationBytes +
    targetWeightBytes +
    RUNTIME_OVERHEAD_BYTES

  // The same peak with the target dropped, which is what the offline mode costs.
  const offlinePeakBytes = peakVramBytes - targetWeightBytes
  const gpusNeeded = Math.max(1, Math.ceil(peakVramBytes / vramBytes))

  let verdict: DsparkVerdict
  if (peakVramBytes <= vramBytes) {
    verdict = 'fits'
  } else if (!offline && offlinePeakBytes <= vramBytes) {
    // The drafter itself fits; it is the resident target that does not.
    verdict = 'needs-offline'
  } else if (gpusNeeded <= MAX_SUGGESTED_GPUS) {
    verdict = 'needs-more-gpus'
  } else {
    verdict = 'needs-offload'
  }

  // --- Explanations -------------------------------------------------------

  const tokensPerEpoch = inputs.trainingTokens.toLocaleString('en-US')

  const steps = [
    {
      label: 'Cache width per token',
      detail: `${inputs.numTargetLayers} captured layers of ${formatExact(hiddenSize)} values in bf16 is ${formatExact(inputs.numTargetLayers * hiddenSize * BF16_BYTES)} bytes, plus ${formatExact(hiddenSize * BF16_BYTES)} for the last hidden state the distribution loss needs, plus the token ids and the two masks. That is ${formatExact(cacheBytesPerToken)} bytes per token.`,
    },
    {
      label: offline ? 'Target cache' : 'No target cache',
      detail: offline
        ? `${formatExact(cacheBytesPerToken)} bytes per token times ${tokensPerEpoch} tokens is ${formatBytes(cacheBytes).text}. This is written once and read back every epoch.`
        : 'Online capture keeps the target resident and writes nothing, so the run costs no storage. It costs the target model in VRAM instead.',
    },
    {
      label: 'Draft parameters',
      detail: `${inputs.numDraftLayers} blocks of backbone is ${formatExact(draftBackboneParams)} parameters, the projection from the captured layers is ${formatExact(draftProjectionParams)}, the rank ${inputs.markovRank} Markov head is ${formatExact(draftMarkovParams)}, and the confidence head is ${formatExact(draftConfidenceParams)}. The embedding and the language model head are shared with the target and frozen, so they are not trained.`,
    },
    {
      label: 'Positions scored',
      detail: `${tokensPerEpoch} tokens at ${formatExact(inputs.sequenceLength)} per sequence is ${formatExact(sequencesPerEpoch)} sequences. Each contributes ${inputs.numAnchors} anchors of ${inputs.blockSize} tokens, so one epoch scores ${formatExact(positionsPerEpoch)} positions, and ${inputs.epochs} epochs score ${formatExact(positionsPerEpoch * inputs.epochs)}.`,
    },
    {
      label: 'Arithmetic',
      detail: `Six flops per parameter per position is ${trainingFlops.toExponential(3)}, the context each block attends to adds ${contextFlops.toExponential(3)}, and preparing the cache adds ${cachePrepFlops.toExponential(3)}. Together, ${totalFlops.toExponential(3)} FLOPs.`,
    },
    {
      label: 'Compute time',
      detail: `${inputs.gpu.bf16DenseTflops.toLocaleString('en-US')} TFLOPS dense on ${inputs.gpu.label}, times ${inputs.gpuCount} card${inputs.gpuCount === 1 ? '' : 's'}, at ${(inputs.mfu * 100).toFixed(0)} percent utilisation, is ${(computeSeconds / 3600).toFixed(2)} hours.`,
    },
    {
      label: 'Cache read time',
      detail: offline
        ? `${formatBytes(cacheBytes).text} read back ${inputs.epochs} times at ${inputs.storage.bandwidthGBs} GB/s is ${(cacheReadSeconds / 3600).toFixed(2)} hours.`
        : 'Online capture reads nothing back, so there is no stream cost and the run is bound by compute.',
    },
    {
      label: 'Estimate',
      detail: `The slower of the two is the ${bound === 'compute' ? 'compute' : 'cache read'} bound, so ${(estimateSeconds / 3600).toFixed(2)} hours after the ${inputs.overheadFactor} times overhead and ${(inputs.setupSeconds / 60).toFixed(0)} minutes of setup.`,
    },
    {
      label: 'Memory',
      detail: `${formatBytes(draftWeightBytes).text} of bf16 weights, ${formatBytes(optimizerBytes).text} of optimizer state, ${formatBytes(gradientBytes).text} of gradients, a ${formatBytes(activationBytes).text} activation buffer, and ${formatBytes(RUNTIME_OVERHEAD_BYTES).text} of framework reserve${offline ? '' : `, plus ${formatBytes(targetWeightBytes).text} for the resident target`}. That is ${formatBytes(peakVramBytes).text} against ${formatBytes(vramBytes).text} on the card.`,
    },
  ]

  const constants = [
    { key: 'target_model_type', value: shape.modelType, source: 'config' },
    { key: 'hidden_size', value: shape.hiddenSize, source: 'config' },
    { key: 'num_hidden_layers', value: shape.numLayers, source: 'config' },
    { key: 'intermediate_size', value: shape.intermediateSize, source: 'config' },
    { key: 'vocab_size', value: shape.vocabSize, source: 'config' },
    {
      key: 'attention_params_per_layer',
      value: shape.attentionParamsPerLayer,
      source: 'derived from the attention layout',
    },
    { key: 'total_target_params', value: shape.totalParams, source: 'sum over every block, including embeddings' },
    {
      key: 'active_target_params_per_token',
      value: shape.activeParamsPerToken,
      source: 'the target parameters one token touches, which set the cache prep cost',
    },
    { key: 'num_target_layers', value: inputs.numTargetLayers, source: 'recipe: captured layers' },
    { key: 'num_draft_layers', value: inputs.numDraftLayers, source: 'recipe: draft backbone depth' },
    { key: 'block_size', value: inputs.blockSize, source: 'recipe: gamma, the drafted block' },
    { key: 'num_anchors', value: inputs.numAnchors, source: 'recipe: blocks sampled per sequence' },
    { key: 'markov_rank', value: inputs.markovRank, source: 'recipe: rank of the sequential head' },
    { key: 'training_tokens', value: inputs.trainingTokens, source: 'one pass over the training set' },
    { key: 'epochs', value: inputs.epochs, source: 'passes over the training set' },
    { key: 'draft_params', value: draftParams, source: 'the trained drafter, excluding the frozen shared embedding and head' },
  ]

  const assumptions = [
    'The drafter shares and freezes the target embedding and language model head, so neither is trained and neither appears in the draft parameter count. This is what the paper and both published implementations do.',
    'Each draft block attends to the target context before its anchor and to itself bidirectionally, so the context term scales with the sequence length. It is a smaller term than the block arithmetic at every setting the presets cover.',
    'Every position in a block is trained in one parallel pass, so the arithmetic is six flops per draft parameter per position rather than the sequence length. This is why a drafter over a billion tokens is affordable at all.',
    'The target cache stores bf16 hidden states, int32 token ids and uint8 masks, matching the layout DeepSpec writes. Storing fewer captured layers reduces the cache in proportion, and is the first lever when the disk is too small.',
    'Cache preparation is one forward pass of the target over the whole training set, so its cost scales with the target rather than the draft. It is included because on a large target it is not negligible.',
    'The optimizer state is two fp32 Adam moments, eight bytes per parameter. The bf16 weights and the gradients are counted separately, so the total is twelve bytes per parameter plus activations.',
    'The activation buffer is an estimate of the live intermediates in the draft forward pass, not a measured figure. Lower the micro batch if the real run runs out of memory.',
    'Model flops utilisation absorbs kernel efficiency. A shallow drafter over short blocks reaches a lower fraction of peak than a large model does, so a third is optimistic and the estimate is the figure to trust.',
    'The overhead factor covers the data loader, the cache reader, the checkpoint writer, and the scheduler, none of which appear in the FLOPs figure.',
    'The confidence head is trained jointly with the draft. Serving its output well needs post-hoc calibration on held-out data, which is not part of this estimate.',
    ...shape.notes,
  ]

  return {
    mode: inputs.dataMode,
    verdict,
    shape,
    cacheBytesPerToken,
    cacheBytes,
    cacheWriteSeconds,
    cacheReadSeconds,
    draftBackboneParams,
    draftProjectionParams,
    draftMarkovParams,
    draftConfidenceParams,
    draftParams,
    draftWeightBytes,
    optimizerBytes,
    gradientBytes,
    trainingTokens: inputs.trainingTokens,
    numTargetLayers: inputs.numTargetLayers,
    blockSize: inputs.blockSize,
    sequencesPerEpoch,
    positionsPerEpoch,
    trainingFlops,
    contextFlops,
    cachePrepFlops,
    totalFlops,
    computeSeconds,
    bound,
    estimateSeconds,
    setupSeconds: inputs.setupSeconds,
    overheadFactor: inputs.overheadFactor,
    activationBytes,
    targetWeightBytes,
    runtimeReserveBytes: RUNTIME_OVERHEAD_BYTES,
    peakVramBytes,
    vramBytes,
    gpusNeeded,
    steps,
    constants,
    assumptions,
  }
}
