/**
 * The reachability engine.
 *
 * For each model on the decision leaderboard this asks the inference engine a
 * single question: at a given workload, which cards in this site's hardware
 * directory can serve the checkpoint, and how fast is the best of them?
 *
 * The sizing is not reimplemented here. detectModelShape reads the checkpoint
 * config, and estimateInference costs the weights, the KV cache, the
 * activation buffer, and the framework reserve against every card. This module
 * only turns one engine result into one leaderboard row and puts the rows in
 * order.
 *
 * A model the site cannot size is kept as a row rather than dropped. A closed
 * hosted API and a config that describes no transformer are both real answers
 * to the reachability question, and hiding them would overstate how many
 * models this page can actually place on hardware.
 */

import { GPU_PRESETS, type GpuSpec } from '../hardware'
import {
  estimateInference,
  type InferenceResult,
  type InferenceVerdict,
} from '../inference'
import type { DtypeId } from '../kvcache'
import { detectModelShape, type ModelShape } from '../model-shape'
import type { RawConfig } from '../model-config'
import type { WeightFormatId } from '../weight-format'

import { REACHABILITY_CONFIGS, REACHABILITY_MODELS } from './dataset.generated'
import type { ReachabilityConfigs, ReachabilityModelRecord } from './types'

/** What a row found, with the two ways of finding nothing kept apart. */
export type ReachabilityVerdict = InferenceVerdict | 'unreachable'

/** The workload every model is costed against. */
export interface ReachabilityInputs {
  /** Tokens in each sequence. */
  contextLength: number
  /** Sequences served at the same time. */
  sequences: number
  /** A format forced on every bucket, or null for the published one. */
  weightFormat: WeightFormatId | null
  /** The most cards the answer may use. */
  maxGpus: number
  /** Share of each card held back for fragmentation and the display. */
  headroom: number
  /** The dtype the KV cache is held in. */
  kvCacheDtype: DtypeId
  /** The cards the ranking may consider. Empty means every card. */
  gpuIds: string[]
}

/** One model, costed against the workload and the hardware directory. */
export interface ReachabilityRow {
  model: ReachabilityModelRecord
  /** False when the model could not be sized at all. */
  sizeable: boolean
  /** Why the model could not be sized, or null when it was. */
  reason: string | null
  /** The shape read from the config, or null when nothing was sized. */
  shape: ModelShape | null
  /** The engine result, which the breakdown panel reads. Null when unsized. */
  result: InferenceResult | null

  /** Bytes for every weight in the checkpoint. */
  weightsBytes: number
  /** Bytes for the KV cache at this workload. */
  kvCacheBytes: number
  /** Every resident byte at this workload, summed over the cards. */
  totalBytes: number

  verdict: ReachabilityVerdict
  /** The configuration the ranking chose, or null when nothing fits. */
  recommendedGpu: GpuSpec | null
  /** Cards in the recommended configuration. */
  gpuCount: number
  /**
   * The smallest single card that holds the whole run, or null when no single
   * card does. This is the card a reader would buy for the model.
   */
  smallestSingleGpu: GpuSpec | null
  /** Cards that hold the run at this workload, out of the considered set. */
  fittingGpuIds: string[]
  fittingGpuCount: number
  /** Cards the ranking considered, which the filter can narrow. */
  consideredGpuCount: number
  /** Estimated decode tokens each second for the recommended configuration. */
  decodeTokensPerSecond: number
  /** The format the checkpoint publishes, as the engine read it. */
  weightFormat: WeightFormatId | null
  /**
   * Cards that hold the whole run on one card. This is the headline reach, and
   * it is the set a reader who wants to host the model on a single card reads.
   */
  singleCardGpuIds: string[]
  singleCardGpuCount: number
  /** Parameter count the leaderboard reports, or null when it lists none. */
  publishedParams: number | null
  /**
   * Parameter count the config arithmetic produced. It is reported beside the
   * published figure because the two can differ, and the reason is stated in
   * the breakdown rather than hidden.
   */
  configParams: number | null
}

/** A row for a model the site could not size. */
function unsizedRow(
  model: ReachabilityModelRecord,
  reason: string,
  consideredGpuCount: number,
): ReachabilityRow {
  return {
    model,
    sizeable: false,
    reason,
    shape: null,
    result: null,
    weightsBytes: 0,
    kvCacheBytes: 0,
    totalBytes: 0,
    verdict: 'unreachable',
    recommendedGpu: null,
    gpuCount: 0,
    smallestSingleGpu: null,
    fittingGpuIds: [],
    fittingGpuCount: 0,
    consideredGpuCount,
    decodeTokensPerSecond: 0,
    weightFormat: null,
    singleCardGpuIds: [],
    singleCardGpuCount: 0,
    publishedParams: model.params,
    configParams: null,
  }
}

/** The model type a config declares, for the reason on an unsized row. */
function modelTypeOf(config: RawConfig): string {
  const inner =
    config.text_config && typeof config.text_config === 'object'
      ? (config.text_config as RawConfig)
      : config
  const type = inner.model_type ?? config.model_type
  return typeof type === 'string' && type !== '' ? type : 'unknown'
}

/**
 * Costs one model against the workload and the hardware directory.
 *
 * A model with no config, a config that describes no transformer, or a shape
 * the engine refuses to size is returned as an unreachable row. Nothing here
 * throws, because one exotic checkpoint must not take the whole table down.
 */
export function evaluateModel(
  model: ReachabilityModelRecord,
  config: RawConfig | null,
  inputs: ReachabilityInputs,
): ReachabilityRow {
  const consideredGpuCount = inputs.gpuIds.length > 0 ? inputs.gpuIds.length : GPU_PRESETS.length

  if (!config) {
    const reason = model.closed
      ? 'This model is served behind a closed hosted API, so there is no checkpoint to size.'
      : `The published config for ${model.baseModel ?? model.name} could not be read, so the model could not be sized.`
    return unsizedRow(model, reason, consideredGpuCount)
  }

  const shape = detectModelShape(config)
  if (!shape) {
    return unsizedRow(
      model,
      `The config for ${model.baseModel ?? model.name} declares the model type "${modelTypeOf(config)}", which is not a decoder transformer this site can size.`,
      consideredGpuCount,
    )
  }

  let result: InferenceResult
  try {
    result = estimateInference(shape, {
      config,
      weightFormat: inputs.weightFormat ?? undefined,
      contextLength: inputs.contextLength,
      sequences: inputs.sequences,
      headroom: inputs.headroom,
      maxGpus: inputs.maxGpus,
      kvCacheDtype: inputs.kvCacheDtype,
      gpuFilter: inputs.gpuIds.length > 0 ? inputs.gpuIds : undefined,
    })
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'The engine cannot size this model.'
    return unsizedRow(model, message, consideredGpuCount)
  }

  // The recommended configuration and the alternatives are exactly the
  // configurations that fit, so the fitting set is those two together.
  const fitting = result.recommended ? [result.recommended, ...result.alternatives] : []
  const fittingGpuIds = fitting.map((candidate) => candidate.gpu.id)

  // The cards that hold the whole run on one card. The engine sorts its
  // candidates fewest cards first, so a single card candidate is the one it
  // recommends whenever one exists.
  const singleCards = fitting
    .filter((candidate) => candidate.gpuCount === 1)
    .sort((a, b) => a.gpu.vramGiB - b.gpu.vramGiB)
  const single = singleCards[0]

  return {
    model,
    sizeable: true,
    reason: null,
    shape,
    result,
    weightsBytes: result.weightsBytes,
    kvCacheBytes: result.kvCacheBytes,
    totalBytes: result.totalBytes,
    verdict: result.verdict,
    recommendedGpu: result.recommended?.gpu ?? null,
    gpuCount: result.recommended?.gpuCount ?? 0,
    smallestSingleGpu: single?.gpu ?? null,
    fittingGpuIds,
    fittingGpuCount: fittingGpuIds.length,
    consideredGpuCount,
    decodeTokensPerSecond: result.decodeTokensPerSecond,
    weightFormat: result.weightFormat,
    singleCardGpuIds: singleCards.map((candidate) => candidate.gpu.id),
    singleCardGpuCount: singleCards.length,
    publishedParams: model.params,
    configParams: shape.totalParams,
  }
}

/** Sort group, so reachable rows come before the ones that are not. */
const GROUP: Record<ReachabilityVerdict, number> = {
  single: 0,
  multi: 1,
  none: 2,
  unreachable: 3,
}

/**
 * The default order: the least hardware first.
 *
 * A model that one small card holds is the most reachable, so the smallest
 * such card decides the order. A model that needs several cards comes next,
 * fewest cards first and then the smallest card. A model that needs more than
 * the card limit comes after that, and a model that could not be sized comes
 * last. Ties break on the decode rate and then on the Decision Index, so the
 * stronger and faster model is the one a reader sees first.
 */
export function compareReachability(a: ReachabilityRow, b: ReachabilityRow): number {
  const group = GROUP[a.verdict] - GROUP[b.verdict]
  if (group !== 0) return group

  if (a.verdict === 'single' && b.verdict === 'single') {
    const vram =
      (a.smallestSingleGpu?.vramGiB ?? Number.POSITIVE_INFINITY) -
      (b.smallestSingleGpu?.vramGiB ?? Number.POSITIVE_INFINITY)
    if (vram !== 0) return vram
  }

  if (a.verdict === 'multi' && b.verdict === 'multi') {
    if (a.gpuCount !== b.gpuCount) return a.gpuCount - b.gpuCount
    const vram = (a.recommendedGpu?.vramGiB ?? 0) - (b.recommendedGpu?.vramGiB ?? 0)
    if (vram !== 0) return vram
  }

  if (a.decodeTokensPerSecond !== b.decodeTokensPerSecond) {
    return b.decodeTokensPerSecond - a.decodeTokensPerSecond
  }

  const index = (b.model.index ?? Number.NEGATIVE_INFINITY) - (a.model.index ?? Number.NEGATIVE_INFINITY)
  if (index !== 0) return index

  return a.model.name.localeCompare(b.model.name)
}

/**
 * Costs every model on the leaderboard and returns the rows in reachability
 * order.
 *
 * The dataset and the config map can be overridden so a test can drive the
 * engine from a small fixture rather than the baked board.
 */
export function rankReachability(
  inputs: ReachabilityInputs,
  models: readonly ReachabilityModelRecord[] = REACHABILITY_MODELS,
  configs: ReachabilityConfigs = REACHABILITY_CONFIGS,
): ReachabilityRow[] {
  return models
    .map((model) => {
      const config = model.configKey ? (configs[model.configKey] ?? null) : null
      return evaluateModel(model, config, inputs)
    })
    .sort(compareReachability)
}
