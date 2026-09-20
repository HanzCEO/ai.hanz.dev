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
    throw new DsparkInputError(`${label} must be a whole number of 1 or more.`, field)
  }
}

function validate(inputs: DsparkInputs): void {
  requirePositive(inputs.trainingTokens, 'trainingTokens', 'Training tokens')
  requirePositive(inputs.epochs, 'epochs', 'Epochs')
  requirePositive(inputs.numAnchors, 'numAnchors', 'Anchors for each sequence')
  requirePositive(inputs.blockSize, 'blockSize', 'Block size')
  requirePositive(inputs.numDraftLayers, 'numDraftLayers', 'Draft layers')
  requirePositive(inputs.numTargetLayers, 'numTargetLayers', 'Captured target layers')
  requirePositive(inputs.sequenceLength, 'sequenceLength', 'Sequence length')
  requirePositive(inputs.gpuCount, 'gpuCount', 'GPU count')
  requirePositive(inputs.microBatchSize, 'microBatchSize', 'Micro batch size')

  if (!Number.isFinite(inputs.markovRank) || inputs.markovRank < 0) {
    throw new DsparkInputError(
      'Markov rank must be 0 or more. A rank of 0 disables the sequential head.',
      'markovRank',
    )
  }
  if (!Number.isFinite(inputs.mfu) || inputs.mfu <= 0 || inputs.mfu > 1) {
    throw new DsparkInputError(
      'Model flops utilisation must be more than 0 and at most 1.',
      'mfu',
    )
  }
  if (!Number.isFinite(inputs.overheadFactor) || inputs.overheadFactor < 1) {
    throw new DsparkInputError('Overhead must be at least 1.', 'overheadFactor')
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
  // sequence times the tokens per block.
  //
  // An anchor count larger than the sequence can hold would score more positions
  // than the sequence has tokens, which cannot happen. It arises when a recipe
  // written for long sequences is pointed at a short one, so the count is capped
  // at one block per sequence token and the cap is reported.
  const maxAnchors = Math.floor(inputs.sequenceLength / inputs.blockSize)
  const numAnchors = Math.max(1, Math.min(inputs.numAnchors, maxAnchors))
  const anchorsClamped = numAnchors < inputs.numAnchors

  const sequencesPerEpoch = inputs.trainingTokens / inputs.sequenceLength
  const positionsPerEpoch = sequencesPerEpoch * numAnchors * inputs.blockSize

  // Six flops per parameter per position: a forward and a backward pass.
  const trainingFlops = 6 * positionsPerEpoch * inputs.epochs * draftParams

  // Each block attends to the target context before its anchor, which is a
  // quadratic term in the context length and does not shrink with the anchors.
  const contextFlops =
    4 *
    numAnchors *
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
    numAnchors *
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
      label: 'Target cache width',
      detail: `The target cache stores ${inputs.numTargetLayers} captured layers of ${formatExact(hiddenSize)} values in bf16. That is ${formatExact(inputs.numTargetLayers * hiddenSize * BF16_BYTES)} bytes. The last hidden state for the distribution loss adds ${formatExact(hiddenSize * BF16_BYTES)} bytes. The token ids and the 2 masks add the rest. One token therefore needs ${formatExact(cacheBytesPerToken)} bytes.`,
    },
    {
      label: offline ? 'Target cache' : 'No target cache',
      detail: offline
        ? `One token needs ${formatExact(cacheBytesPerToken)} bytes, so ${tokensPerEpoch} tokens need ${formatBytes(cacheBytes).text}. The run writes this cache once and reads it back in each epoch.`
        : 'Online capture keeps the target in VRAM and writes nothing. The run therefore needs no storage. It needs the target in VRAM instead.',
    },
    {
      label: 'Drafter parameters',
      detail: `The ${inputs.numDraftLayers} backbone blocks hold ${formatExact(draftBackboneParams)} parameters. The projection from the captured layers holds ${formatExact(draftProjectionParams)}. The Markov head at rank ${inputs.markovRank} holds ${formatExact(draftMarkovParams)}. The confidence head holds ${formatExact(draftConfidenceParams)}. The embedding and the language model head are shared with the target and frozen. The run therefore does not train them.`,
    },
    {
      label: 'Positions scored',
      detail: `The training set holds ${tokensPerEpoch} tokens at ${formatExact(inputs.sequenceLength)} tokens for each sequence. That is ${formatExact(sequencesPerEpoch)} sequences. Each sequence contributes ${formatExact(numAnchors)} anchors of ${inputs.blockSize} tokens. One epoch therefore scores ${formatExact(positionsPerEpoch)} positions, and ${inputs.epochs} epochs score ${formatExact(positionsPerEpoch * inputs.epochs)}.${anchorsClamped ? ` The requested ${formatExact(inputs.numAnchors)} anchors were reduced to ${formatExact(numAnchors)}, because a ${formatExact(inputs.sequenceLength)} token sequence cannot hold more. One block for each sequence token is the maximum.` : ''}`,
    },
    {
      label: 'Arithmetic',
      detail: `6 FLOPs for each parameter and each position gives ${trainingFlops.toExponential(3)}. The context that each block attends to adds ${contextFlops.toExponential(3)}. Preparing the target cache adds ${cachePrepFlops.toExponential(3)}. The total is ${totalFlops.toExponential(3)} FLOPs.`,
    },
    {
      label: 'Arithmetic duration',
      detail: `One ${inputs.gpu.label} reaches ${inputs.gpu.bf16DenseTflops.toLocaleString('en-US')} TFLOPS dense. With ${inputs.gpuCount} GPU at ${(inputs.mfu * 100).toFixed(0)} percent utilisation, the arithmetic takes ${(computeSeconds / 3600).toFixed(2)} hours.`,
    },
    {
      label: 'Target cache reads',
      detail: offline
        ? `The run reads ${formatBytes(cacheBytes).text} back ${inputs.epochs} times from storage at ${inputs.storage.bandwidthGBs} GB/s. That takes ${(cacheReadSeconds / 3600).toFixed(2)} hours.`
        : 'Online capture reads nothing back. The run therefore has no read cost, and the arithmetic sets the bound.',
    },
    {
      label: 'Estimate',
      detail: `The slower value sets the bound, and that bound is ${bound === 'compute' ? 'the arithmetic' : 'the cache read'}. The estimate is ${(estimateSeconds / 3600).toFixed(2)} hours, after ${inputs.overheadFactor} times overhead and ${(inputs.setupSeconds / 60).toFixed(0)} minutes of setup.`,
    },
    {
      label: 'VRAM',
      detail: `The drafter weights need ${formatBytes(draftWeightBytes).text} in bf16. The optimizer state needs ${formatBytes(optimizerBytes).text}. The gradients need ${formatBytes(gradientBytes).text}. The activation buffer needs ${formatBytes(activationBytes).text}. The framework reserve needs ${formatBytes(RUNTIME_OVERHEAD_BYTES).text}.${offline ? '' : ` The resident target needs ${formatBytes(targetWeightBytes).text}.`} The peak is therefore ${formatBytes(peakVramBytes).text}, against ${formatBytes(vramBytes).text} of VRAM on the GPU.`,
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
    { key: 'total_target_params', value: shape.totalParams, source: 'the sum over every block, including the embeddings' },
    {
      key: 'active_target_params_per_token',
      value: shape.activeParamsPerToken,
      source: 'the target parameters that one token touches. These set the cost to prepare the target cache.',
    },
    { key: 'num_target_layers', value: inputs.numTargetLayers, source: 'recipe: the captured layers' },
    { key: 'num_draft_layers', value: inputs.numDraftLayers, source: 'recipe: the depth of the drafter backbone' },
    { key: 'block_size', value: inputs.blockSize, source: 'recipe: gamma, the drafted block' },
    { key: 'num_anchors', value: numAnchors, source: 'recipe: the blocks sampled from each sequence' },
    { key: 'markov_rank', value: inputs.markovRank, source: 'recipe: the rank of the sequential head' },
    { key: 'training_tokens', value: inputs.trainingTokens, source: '1 pass over the training set' },
    { key: 'epochs', value: inputs.epochs, source: 'the number of passes over the training set' },
    { key: 'draft_params', value: draftParams, source: 'the trained drafter, without the frozen shared embedding and head' },
  ]

  const assumptions = [
    'The drafter shares the target embedding and language model head, and the run freezes both. The run does not train either one, and neither appears in the drafter parameter count. The paper and both published implementations do this.',
    'Each drafter block attends to the target context before its anchor, and to itself in both directions. The context term therefore scales with the sequence length. That term is smaller than the block arithmetic at every setting in the recipes.',
    'The run trains every position in a block in 1 parallel pass. The arithmetic is therefore 6 FLOPs for each drafter parameter and each position. It does not depend on the sequence length. This is why a drafter over a billion tokens is affordable.',
    'The target cache stores bf16 hidden states, int32 token ids, and uint8 masks. This matches the layout that DeepSpec writes. Fewer captured layers make the target cache smaller in proportion. That is the first control when the storage is too small.',
    'Target cache preparation is 1 forward pass of the target over the whole training set. Its cost therefore scales with the target and not with the drafter. The estimate includes it, because it is large for a large target.',
    'The optimizer state is 2 fp32 Adam moments, or 8 bytes for each parameter. The calculator counts the bf16 weights and the gradients separately. The total is therefore 12 bytes for each parameter, plus the activations.',
    'The activation buffer is an estimate of the live intermediates in the drafter forward pass. It is not a measured value. Lower the micro batch if the real run runs out of VRAM.',
    'Model flops utilisation covers the kernel efficiency. A shallow drafter over short blocks reaches a smaller share of the peak than a large model does. One third is therefore optimistic. Trust the estimate and not the raw FLOPs.',
    'The overhead factor covers the data loader, the cache reader, the checkpoint writer, and the scheduler. None of these appear in the FLOPs.',
    anchorsClamped
      ? `The run caps the anchor count at ${formatExact(numAnchors)} for each sequence, which is 1 block for each sequence token. The requested ${formatExact(inputs.numAnchors)} anchors would score more positions than a ${formatExact(inputs.sequenceLength)} token sequence holds. Raise the sequence length or lower the anchor count to change this.`
      : 'The anchor count fits the sequence length. The run therefore scores every requested block.',
    'The run trains the confidence head together with the drafter. Good output at serving time needs post-hoc scaling on held-out data. That work is not part of this estimate.',
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
    numAnchors,
    anchorsClamped,
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
