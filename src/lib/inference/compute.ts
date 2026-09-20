import { formatBytes, formatExact } from '../format'
import { GPU_PRESETS, type GpuSpec } from '../hardware'
import { computeKvCache } from '../kvcache'
import type { ModelShape } from '../model-shape'

import {
  ACTIVATION_BYTES_PER_ELEMENT,
  ACTIVATION_FACTOR,
  BANDWIDTH_EFFICIENCY,
  BYTES_PER_WEIGHT,
  DEFAULT_MTP_HEAD,
  GIB,
  PRECISIONS,
  RUNTIME_OVERHEAD_BYTES,
  isMtpHeadType,
  mtpHeadSpec,
  mtpSpeedup,
} from './presets'
import {
  InferenceInputError,
  type InferenceCandidate,
  type InferenceInputField,
  type InferenceInputs,
  type InferenceResult,
  type InferenceVerdict,
} from './types'

function requirePositive(value: number, field: InferenceInputField, label: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new InferenceInputError(`${label} must be a whole number of 1 or more.`, field)
  }
}

function validate(inputs: InferenceInputs): void {
  if (!PRECISIONS.includes(inputs.precision as (typeof PRECISIONS)[number])) {
    throw new InferenceInputError(
      `Unknown precision "${String(inputs.precision)}". Pick FP16 or BF16.`,
      'precision',
    )
  }
  if (inputs.mtpHead !== undefined && !isMtpHeadType(inputs.mtpHead)) {
    throw new InferenceInputError(
      `Unknown MTP head "${String(inputs.mtpHead)}". Pick a head from the list.`,
      'mtpHead',
    )
  }
  requirePositive(inputs.contextLength, 'contextLength', 'Context length')
  requirePositive(inputs.sequences, 'sequences', 'Concurrent sequences')
  requirePositive(inputs.maxGpus, 'maxGpus', 'Maximum GPU count')

  if (!Number.isFinite(inputs.headroom) || inputs.headroom < 0 || inputs.headroom >= 1) {
    throw new InferenceInputError(
      'The VRAM headroom must be at least 0 percent and less than 100 percent.',
      'headroom',
    )
  }
}

/**
 * Estimates the GPU configuration a model needs for FP16 or BF16 inference.
 *
 * The three questions are how much memory the run needs, which card holds it,
 * and how fast the answer comes out. All three follow from the model config,
 * the context length, and the number of sequences served at once.
 *
 * Two facts drive the whole model. First, FP16 and BF16 both take two bytes for
 * each weight, so the footprint is the same for either one. Second, the weights
 * and the KV cache divide across the tensor-parallel ranks while the activation
 * buffer and the framework reserve stay in full on every card. A second card
 * therefore does not halve the footprint, which is why the smallest
 * configuration is not always the one with the fewest cards.
 *
 * Decode is bound by memory bandwidth rather than by arithmetic. Each token
 * reads every active weight once, so the throughput figure here is a bandwidth
 * roofline. It ignores prefill, kernel launch overhead, and the cost of the
 * interconnect that tensor parallelism needs.
 *
 * A speculative decoding head raises that roofline, because one pass of the
 * model then yields several accepted tokens. The head is a property of the
 * trained checkpoint, so the multiplier is an estimate for measurement rather
 * than a promise. It moves the decode rate and leaves every memory term alone.
 */
export function estimateInference(shape: ModelShape, inputs: InferenceInputs): InferenceResult {
  validate(inputs)

  const { precision, contextLength, sequences, headroom, maxGpus } = inputs
  const bytesPerWeight = BYTES_PER_WEIGHT

  // A head is trained against one model, so it is a property of the checkpoint
  // and not of the hardware. An absent head is no head, which is the roofline.
  const mtpHead = inputs.mtpHead ?? DEFAULT_MTP_HEAD
  const mtpMultiplier = mtpSpeedup(mtpHead)

  // --- Memory -------------------------------------------------------------
  //
  // Every weight in the checkpoint has to be resident, including the shared
  // experts, the router, and the language model head.
  const weightsBytes = shape.totalParams * bytesPerWeight

  // The KV cache shape is read from the config by the shared engine, so a
  // grouped query model, a latent attention model, and a hybrid linear model
  // each get their own formula rather than one approximation. The cache dtype
  // is the cache layer's choice and defaults to the weight precision.
  const kvCacheDtype = inputs.kvCacheDtype ?? precision
  const indexerDtype = inputs.indexerDtype ?? precision
  const kv = computeKvCache(inputs.config, {
    contextLength,
    sequenceCount: sequences,
    kvCacheDtype,
    indexerDtype,
  })
  const kvCacheBytes = kv.totalBytes
  const kvBytesPerToken = kv.bytesPerToken

  // The activation buffer is sized on the widest live tensor of a decode step.
  // It stays in full on every card, because each rank holds its own slice of
  // the batch rather than a slice of the buffer.
  const activationBytes =
    sequences * contextLength * shape.hiddenSize * ACTIVATION_BYTES_PER_ELEMENT * ACTIVATION_FACTOR
  const runtimeReserveBytes = RUNTIME_OVERHEAD_BYTES
  const totalBytes = weightsBytes + kvCacheBytes + activationBytes + runtimeReserveBytes

  // --- The recommendation ------------------------------------------------
  const gpus = inputs.gpuFilter
    ? GPU_PRESETS.filter((gpu) => inputs.gpuFilter?.includes(gpu.id))
    : GPU_PRESETS

  const divisibleBytes = weightsBytes + kvCacheBytes
  const undividedBytes = activationBytes + runtimeReserveBytes
  const perCardBytesFor = (cards: number) => divisibleBytes / cards + undividedBytes
  const usableBytesFor = (gpu: GpuSpec) => gpu.vramGiB * GIB * (1 - headroom)

  // Each token reads every active weight and the whole KV cache once. A head
  // raises the rate, because one pass of the model then yields several
  // accepted tokens. The memory terms above do not move with it.
  const stepBytes = shape.activeParamsPerToken * bytesPerWeight + kvCacheBytes
  const decodeTokensPerSecondFor = (gpu: GpuSpec, cards: number) =>
    stepBytes > 0
      ? ((gpu.bandwidthGBs * 1e9 * cards * BANDWIDTH_EFFICIENCY) / stepBytes) *
        sequences *
        mtpMultiplier
      : 0

  const build = (gpu: GpuSpec, cards: number, fits: boolean): InferenceCandidate => {
    const perCardBytes = perCardBytesFor(cards)
    const usableBytes = usableBytesFor(gpu)
    return {
      gpu,
      gpuCount: cards,
      perCardBytes,
      usableBytes,
      headroomBytes: Math.max(0, usableBytes - perCardBytes),
      decodeTokensPerSecond: decodeTokensPerSecondFor(gpu, cards),
      fits,
    }
  }

  const candidates: InferenceCandidate[] = []
  const misses: InferenceCandidate[] = []

  for (const gpu of gpus) {
    // The per-card footprint falls as the card count rises, so the first count
    // that fits is the smallest one.
    let found: number | null = null
    for (let cards = 1; cards <= maxGpus; cards += 1) {
      if (perCardBytesFor(cards) <= usableBytesFor(gpu)) {
        found = cards
        break
      }
    }
    if (found === null) misses.push(build(gpu, maxGpus, false))
    else candidates.push(build(gpu, found, true))
  }

  // Fewest cards first, then the smallest card that holds them, then the
  // fastest. A smaller card is the cheaper answer, so it wins a tie on size.
  candidates.sort((a, b) => {
    if (a.gpuCount !== b.gpuCount) return a.gpuCount - b.gpuCount
    if (a.gpu.vramGiB !== b.gpu.vramGiB) return a.gpu.vramGiB - b.gpu.vramGiB
    return b.decodeTokensPerSecond - a.decodeTokensPerSecond
  })

  const recommended = candidates[0] ?? null
  const alternatives = candidates.slice(1)

  // When nothing fits, the nearest miss is the configuration that comes closest
  // to the card, which is what the page names in its alert.
  const closest =
    recommended === null
      ? misses.reduce<InferenceCandidate | null>((best, miss) => {
          if (!best) return miss
          return miss.perCardBytes / miss.usableBytes < best.perCardBytes / best.usableBytes
            ? miss
            : best
        }, null)
      : null

  const verdict: InferenceVerdict =
    recommended === null ? 'none' : recommended.gpuCount === 1 ? 'single' : 'multi'

  // --- Throughput ---------------------------------------------------------
  const effectiveBandwidth = recommended
    ? recommended.gpu.bandwidthGBs * 1e9 * recommended.gpuCount * BANDWIDTH_EFFICIENCY
    : 0
  const perSequenceTokensPerSecond =
    recommended && stepBytes > 0 ? (effectiveBandwidth / stepBytes) * mtpMultiplier : 0
  const decodeTokensPerSecond = recommended?.decodeTokensPerSecond ?? 0

  // The roofline the head was applied to, so the page can name what the head
  // added. A rate of zero has no head to divide out, so it stays zero.
  const baseDecodeTokensPerSecond =
    mtpMultiplier > 0 ? decodeTokensPerSecond / mtpMultiplier : 0
  const basePerSequenceTokensPerSecond =
    mtpMultiplier > 0 ? perSequenceTokensPerSecond / mtpMultiplier : 0

  // --- Room to grow -------------------------------------------------------
  //
  // The KV cache is the only term that grows with the context and the sequence
  // count, so the free VRAM converts into one of the two directly.
  const kvHeadroomBytes = recommended?.headroomBytes ?? 0
  const maxContextAtSequences =
    recommended && kvBytesPerToken > 0
      ? contextLength + Math.floor(kvHeadroomBytes / (kvBytesPerToken * sequences))
      : null
  const maxSequencesAtContext =
    recommended && kvBytesPerToken > 0
      ? sequences + Math.floor(kvHeadroomBytes / (kvBytesPerToken * contextLength))
      : null

  // --- Explanations -------------------------------------------------------

  const recommendedLabel = recommended
    ? `${recommended.gpuCount} x ${recommended.gpu.label}`
    : 'no configuration within the limit'

  const steps: Array<{ label: string; detail: string }> = [
    {
      label: 'Resident weights',
      detail: `${shape.modelType} holds ${formatExact(shape.totalParams)} parameters. At ${bytesPerWeight} bytes for each weight in ${precision}, the weights need ${formatBytes(weightsBytes).text}. Both FP16 and BF16 are two bytes, so the other precision gives the same figure.`,
    },
    {
      label: 'KV cache',
      detail: `${kv.architecture.label}. At ${formatExact(contextLength)} tokens for each of ${formatExact(sequences)} ${sequences === 1 ? 'sequence' : 'sequences'}, one token in one sequence needs ${formatExact(kvBytesPerToken)} bytes. The whole cache needs ${formatBytes(kvCacheBytes).text} in ${kvCacheDtype}.`,
    },
    {
      label: 'Activation buffer',
      detail: `A decode step keeps ${ACTIVATION_FACTOR} live intermediates of ${formatExact(sequences * contextLength * shape.hiddenSize)} values at ${ACTIVATION_BYTES_PER_ELEMENT} bytes each. That is ${formatBytes(activationBytes).text}. This buffer stays in full on every card.`,
    },
    {
      label: 'Framework reserve',
      detail: `The CUDA context, the framework, and the resident kernels need ${formatBytes(runtimeReserveBytes).text}. This also stays in full on every card.`,
    },
    {
      label: 'Total',
      detail: `The weights, the cache, the buffer, and the reserve come to ${formatBytes(totalBytes).text} in total.`,
    },
    {
      label: 'Card budget',
      detail: `With ${(headroom * 100).toFixed(0)} percent held back for fragmentation, a card offers ${(100 - headroom * 100).toFixed(0)} percent of its VRAM. The ranking holds back that share on every card.`,
    },
    {
      label: 'Recommendation',
      detail: `The weights and the cache divide across the tensor-parallel ranks. The buffer and the reserve do not. The smallest configuration that fits is ${recommendedLabel}.${recommended ? ` One card then holds ${formatBytes(recommended.perCardBytes).text} against ${formatBytes(recommended.usableBytes).text} of usable VRAM.` : ` No card within the limit holds it at ${formatExact(maxGpus)} ${maxGpus === 1 ? 'card' : 'cards'}.`}`,
    },
    {
      label: 'Decode throughput',
      detail: `Each token reads ${formatExact(shape.activeParamsPerToken)} active parameters and the whole cache, which is ${formatBytes(stepBytes).text}.${recommended ? ` ${recommended.gpuCount} x ${recommended.gpu.label} offers ${formatExact(recommended.gpu.bandwidthGBs * recommended.gpuCount)} GB/s at ${(BANDWIDTH_EFFICIENCY * 100).toFixed(0)} percent of the peak. That roofline gives about ${formatExact(basePerSequenceTokensPerSecond)} tokens each second for one sequence.` : ''}`,
    },
    {
      label: 'MTP head',
      detail:
        mtpHead === DEFAULT_MTP_HEAD
          ? 'No speculative decoding head is selected, so the answer stays on the bandwidth roofline. The head is trained against one model, so the figure is an estimate for measurement.'
          : `${mtpHeadSpec(mtpHead)?.label ?? mtpHead} applies a ${mtpMultiplier}x multiplier to the roofline, which gives about ${formatExact(perSequenceTokensPerSecond)} tokens each second for one sequence. The head is trained against one model, so the figure is an estimate for measurement and not a guarantee.`,
    },
    {
      label: 'Room to grow',
      detail: recommended
        ? `The recommendation leaves ${formatBytes(kvHeadroomBytes).text} free on one card. The cache is the only term that grows.${maxContextAtSequences !== null ? ` That room holds ${formatExact(maxContextAtSequences)} tokens of context at ${formatExact(sequences)} ${sequences === 1 ? 'sequence' : 'sequences'}.` : ''}${maxSequencesAtContext !== null ? ` It also holds ${formatExact(maxSequencesAtContext)} sequences at ${formatExact(contextLength)} tokens.` : ''}`
        : 'There is no room to report, because no configuration fits.',
    },
  ]

  const constants: Array<{ key: string; value: number | string; source: string }> = [
    { key: 'model_type', value: shape.modelType, source: 'config' },
    { key: 'hidden_size', value: shape.hiddenSize, source: 'config' },
    { key: 'num_hidden_layers', value: shape.numLayers, source: 'config' },
    { key: 'intermediate_size', value: shape.intermediateSize, source: 'config' },
    { key: 'vocab_size', value: shape.vocabSize, source: 'config' },
    { key: 'tie_word_embeddings', value: String(shape.tiedEmbeddings), source: 'config' },
    {
      key: 'total_params',
      value: shape.totalParams,
      source: 'the sum over every block, including the embedding, the head, the router and the shared experts',
    },
    {
      key: 'active_params_per_token',
      value: shape.activeParamsPerToken,
      source: 'the weights one token reads on the forward pass, router excluded. These set the decode bandwidth.',
    },
    { key: 'precision', value: precision, source: 'the precision the model is served in' },
    {
      key: 'mtp_head',
      value: mtpHead,
      source: 'the speculative decoding head the model is served with, which is a property of the checkpoint',
    },
    {
      key: 'mtp_speedup',
      value: mtpMultiplier,
      source: 'the multiplier the head applies to the decode rate, held below the published figure',
    },
    { key: 'bytes_per_weight', value: bytesPerWeight, source: 'FP16 and BF16 are both 2 bytes' },
    {
      key: 'kv_cache_dtype',
      value: kvCacheDtype,
      source: 'the dtype the cache layer holds the KV cache in',
    },
    {
      key: 'indexer_dtype',
      value: indexerDtype,
      source: 'the dtype the cache layer holds a sparse indexer cache in',
    },
    { key: 'context_length', value: contextLength, source: 'tokens in each sequence' },
    { key: 'sequences', value: sequences, source: 'sequences served at the same time' },
    { key: 'kv_bytes_per_token', value: kvBytesPerToken, source: `the ${kv.architecture.label} cache shape, for one token in one sequence` },
    { key: 'vram_headroom', value: `${(headroom * 100).toFixed(0)} percent`, source: 'the share of each card held back' },
    { key: 'max_gpus', value: maxGpus, source: 'the most cards the answer may use' },
    { key: 'runtime_reserve_bytes', value: runtimeReserveBytes, source: 'the framework, the CUDA context and the resident kernels' },
  ]

  if (recommended) {
    constants.push(
      { key: 'recommended_gpu', value: recommended.gpu.label, source: 'the smallest card that holds the run' },
      { key: 'recommended_gpu_count', value: recommended.gpuCount, source: 'the fewest cards that hold the run' },
      { key: 'recommended_bandwidth_gbs', value: recommended.gpu.bandwidthGBs, source: 'the memory bandwidth of one card' },
    )
  }

  const assumptions = [
    'FP16 and BF16 are both two bytes for each weight. A model served in either one needs the same VRAM and the same bandwidth. The two differ in numeric range and not in size.',
    'The weights and the KV cache divide across the tensor-parallel ranks. The activation buffer and the framework reserve stay in full on every card. A second card therefore does not halve the footprint.',
    'The activation buffer is an estimate of the live intermediates in a decode step. It is not a measured value. Lower the sequence count if the real run runs out of VRAM.',
    `Decode is bound by memory bandwidth, because each token reads every active weight once. The throughput figure is a bandwidth roofline at ${(BANDWIDTH_EFFICIENCY * 100).toFixed(0)} percent of the peak. It ignores prefill and kernel launch overhead.`,
    'Tensor parallelism across more than one node needs a fast interconnect. This estimate assumes the cards share the work at the bandwidth quoted. A PCIe link between nodes cannot do that.',
    'The parameter count comes from the config. Norms and biases are left out, because they are a fraction of a percent of the weights.',
    'The model is served in FP16 or BF16 with no quantization. A quantized checkpoint is smaller and needs less hardware.',
    'The KV cache figure is the logical size of the cache. A serving engine adds its own overhead on top, for example page padding or per block alignment.',
    'The KV cache dtype is set in the cache layer and not by the weight precision. A narrower cache dtype lowers the cache size and therefore the VRAM. It never lowers the weight size, because the weights are still served in FP16 or BF16.',
    'MTP is a property of the trained checkpoint. A head trained for one model does not transfer to another. The multiplier here is an estimate for measurement.',
    ...kv.assumptions,
  ]

  const notes = [...shape.notes]
  if (kv.bestEffort) {
    notes.push(
      'The KV cache shape had to be inferred from the config fields. Check the cache figure before sizing hardware on it.',
    )
  }

  return {
    shape,
    precision,
    bytesPerWeight,
    kvCacheDtype,
    indexerDtype,
    contextLength,
    sequences,
    maxGpus,
    weightsBytes,
    kvCacheBytes,
    kvBytesPerToken,
    activationBytes,
    runtimeReserveBytes,
    totalBytes,
    verdict,
    recommended,
    alternatives,
    closest,
    mtpHead,
    mtpSpeedup: mtpMultiplier,
    decodeTokensPerSecond,
    perSequenceTokensPerSecond,
    baseDecodeTokensPerSecond,
    basePerSequenceTokensPerSecond,
    kvHeadroomBytes,
    maxContextAtSequences,
    maxSequencesAtContext,
    steps,
    constants,
    assumptions,
    bestEffort: shape.bestEffort || kv.bestEffort,
    notes,
  }
}
