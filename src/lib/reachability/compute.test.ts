import { describe, expect, it } from 'vitest'

import type { RawConfig } from '@/lib/model-config'

import { evaluateModel, rankReachability } from './compute'
import { DEFAULT_REACHABILITY_CONTEXT } from './presets'
import { REACHABILITY_MODELS } from './dataset.generated'
import type { ReachabilityInputs } from './compute'
import type { ReachabilityModelRecord } from './types'

/**
 * The engine is exercised from small hand written configs rather than from the
 * baked board, so a test states its own model and does not move when the
 * upstream leaderboard is refreshed. The arithmetic of each fixture is checked
 * against a named card below, which is what makes the expected card an
 * assertion about the engine rather than about the dataset.
 */

/** A model small enough that every card in the directory holds it. */
const SMALL_MODEL: RawConfig = {
  model_type: 'qwen3',
  hidden_size: 512,
  num_hidden_layers: 4,
  intermediate_size: 1024,
  vocab_size: 2048,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  head_dim: 64,
  tie_word_embeddings: true,
  torch_dtype: 'bfloat16',
  max_position_embeddings: 40960,
}

/**
 * A model that does not fit on a 24 GB card in BF16 but does fit on a 32 GB
 * one, and that needs several 12 GB cards when only those are considered.
 */
const LARGE_MODEL: RawConfig = {
  model_type: 'qwen3',
  hidden_size: 4096,
  num_hidden_layers: 48,
  intermediate_size: 14336,
  vocab_size: 128256,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  head_dim: 128,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
  max_position_embeddings: 262144,
}

/** The number the leaderboard reports for the large fixture. */
const LARGE_PARAMS = 11_500_000_000

function record(
  engine: string,
  configKey: string | null,
  params: number | null = null,
): ReachabilityModelRecord {
  return {
    engine,
    name: engine,
    org: null,
    kind: 'full fine-tune',
    baseModel: configKey,
    modelUrl: null,
    params,
    index: 50,
    latencyMedian: null,
    latencyP95: null,
    closed: false,
    source: 'community',
    panelCoverage: 38,
    missing: [],
    formatHint: null,
    configKey,
  }
}

const BASE: ReachabilityInputs = {
  contextLength: DEFAULT_REACHABILITY_CONTEXT,
  sequences: 1,
  weightFormat: null,
  maxGpus: 8,
  headroom: 0.1,
  kvCacheDtype: 'BF16',
  gpuIds: [],
}

describe('evaluateModel', () => {
  it('places a small model on a single card and reports every card', () => {
    const row = evaluateModel(record('small', 'small'), SMALL_MODEL, BASE)

    expect(row.sizeable).toBe(true)
    expect(row.verdict).toBe('single')
    expect(row.gpuCount).toBe(1)
    // The smallest card in the directory is 8 GB, and several cards carry it.
    expect(row.smallestSingleGpu?.vramGiB).toBe(8)
    expect(row.singleCardGpuCount).toBe(row.consideredGpuCount)
    expect(row.decodeTokensPerSecond).toBeGreaterThan(0)
  })

  it('needs several small cards when the card limit is one and the card is small', () => {
    const multi = evaluateModel(record('large', 'large'), LARGE_MODEL, {
      ...BASE,
      gpuIds: ['rtx-3060'],
    })
    expect(multi.verdict).toBe('multi')
    expect(multi.gpuCount).toBeGreaterThan(1)
    // No 12 GB card holds the whole run on its own.
    expect(multi.singleCardGpuCount).toBe(0)
    expect(multi.smallestSingleGpu).toBeNull()

    const none = evaluateModel(record('large', 'large'), LARGE_MODEL, {
      ...BASE,
      gpuIds: ['rtx-3060'],
      maxGpus: 1,
    })
    expect(none.verdict).toBe('none')
    expect(none.fittingGpuCount).toBe(0)
    expect(none.recommendedGpu).toBeNull()
  })

  it('marks a model with no config unreachable and says why', () => {
    const row = evaluateModel(record('closed', null), null, BASE)

    expect(row.sizeable).toBe(false)
    expect(row.verdict).toBe('unreachable')
    expect(row.reason).toBeTruthy()
    // The table has room for a few words, so the reason is carried twice: the
    // sentence for the breakdown and the label for the row.
    expect(row.reasonShort).toBeTruthy()
    expect(row.result).toBeNull()
    expect(row.singleCardGpuCount).toBe(0)
  })

  it('labels a closed hosted API as closed rather than as a missing config', () => {
    const row = evaluateModel({ ...record('closed', null), closed: true }, null, BASE)

    expect(row.reasonShort).toBe('Closed hosted API')
  })

  it('marks a config that describes no transformer unreachable', () => {
    const extractor: RawConfig = { model_type: 'extractor', max_width: 8 }
    const row = evaluateModel(record('extractor', 'extractor'), extractor, BASE)

    expect(row.sizeable).toBe(false)
    expect(row.verdict).toBe('unreachable')
    expect(row.reason).toContain('extractor')
    // The label names the declared type, because "not a decoder transformer"
    // alone does not tell a reader which architecture was found instead.
    expect(row.reasonShort).toContain('extractor')
  })

  it('leaves the short reason empty on a row that was sized', () => {
    const row = evaluateModel(record('small', 'small'), SMALL_MODEL, BASE)

    expect(row.sizeable).toBe(true)
    expect(row.reason).toBeNull()
    expect(row.reasonShort).toBeNull()
  })

  it('reaches more cards when a narrower weight format is forced', () => {
    const gpuIds = ['rtx-4090', 'rtx-5090']
    const bf16 = evaluateModel(record('large', 'large'), LARGE_MODEL, { ...BASE, gpuIds })
    const mxfp4 = evaluateModel(record('large', 'large'), LARGE_MODEL, {
      ...BASE,
      gpuIds,
      weightFormat: 'MXFP4',
    })

    // In BF16 only the 32 GB card holds the run. In MXFP4 the 24 GB card does
    // too, because the checkpoint costs a quarter of the bytes.
    expect(bf16.singleCardGpuCount).toBe(1)
    expect(bf16.smallestSingleGpu?.id).toBe('rtx-5090')
    expect(mxfp4.singleCardGpuCount).toBe(2)
    expect(mxfp4.smallestSingleGpu?.id).toBe('rtx-4090')
    expect(mxfp4.weightsBytes).toBeLessThan(bf16.weightsBytes)
  })

  it('changes fit when the context length grows', () => {
    const gpuIds = ['rtx-5090']
    const short = evaluateModel(record('large', 'large'), LARGE_MODEL, { ...BASE, gpuIds })
    const long = evaluateModel(record('large', 'large'), LARGE_MODEL, {
      ...BASE,
      gpuIds,
      contextLength: 131072,
    })

    expect(short.verdict).toBe('single')
    expect(short.singleCardGpuCount).toBe(1)
    // The weights are unchanged. The cache is not, and it pushes the run off
    // the one card that held it at the short context.
    expect(long.weightsBytes).toBe(short.weightsBytes)
    expect(long.kvCacheBytes).toBeGreaterThan(short.kvCacheBytes)
    expect(long.verdict).not.toBe('single')
  })

  it('reports the published parameter count beside the config count', () => {
    const row = evaluateModel(record('large', 'large', LARGE_PARAMS), LARGE_MODEL, BASE)
    expect(row.publishedParams).toBe(LARGE_PARAMS)
    expect(row.configParams).toBeGreaterThan(0)
  })
})

describe('rankReachability', () => {
  const rows = rankReachability(BASE)

  it('returns one row per model on the board', () => {
    expect(rows).toHaveLength(REACHABILITY_MODELS.length)
  })

  it('orders every reachable row before the unreachable ones', () => {
    const firstUnreachable = rows.findIndex((row) => row.verdict === 'unreachable')
    expect(firstUnreachable).toBeGreaterThan(-1)
    for (const row of rows.slice(firstUnreachable)) {
      expect(row.verdict).toBe('unreachable')
    }
  })

  it('puts the models that one small card holds first', () => {
    const reachable = rows.filter((row) => row.verdict === 'single')
    expect(reachable.length).toBeGreaterThan(0)
    for (let index = 1; index < reachable.length; index += 1) {
      const previous = reachable[index - 1].smallestSingleGpu?.vramGiB ?? 0
      const current = reachable[index].smallestSingleGpu?.vramGiB ?? 0
      expect(current).toBeGreaterThanOrEqual(previous)
    }
  })

  it('ranks a single card answer ahead of a multi card one, and both ahead of nothing', () => {
    const models = [
      record('big', 'large', LARGE_PARAMS),
      record('closed', null),
      record('small', 'small'),
    ]
    const configs = { small: SMALL_MODEL, large: LARGE_MODEL }
    const ordered = rankReachability({ ...BASE, gpuIds: ['rtx-3060'] }, models, configs)

    expect(ordered.map((row) => row.model.engine)).toEqual(['small', 'big', 'closed'])
    expect(ordered.map((row) => row.verdict)).toEqual(['single', 'multi', 'unreachable'])
  })
})
