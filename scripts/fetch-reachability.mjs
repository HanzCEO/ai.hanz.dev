/**
 * Bakes the zero-shot classification reachability dataset.
 *
 * The reachability leaderboard lists the models on the clef-evals Decision
 * Model Leaderboard and sizes each one against the GPUs in src/lib/hardware.
 * Two things have to be read from the network to do that:
 *
 *   1. The leaderboard document itself, which carries the model list, the
 *      Decision Index, and the base checkpoint each model was built on.
 *   2. The published HuggingFace config of that base checkpoint, which is what
 *      the sizing engine reads for the layer count, the widths, and the
 *      attention shape.
 *
 * Neither is fetched in the browser. The leaderboard sends no CORS header, so
 * a page on ai.hanz.dev could not read it, and the site is a static build that
 * has to render its first paint without a network round trip. This script
 * fetches both once and writes them into a committed module, so a build is
 * offline, deterministic, and cannot change under the page between releases.
 *
 * Run it with `pnpm data:reachability`. A model whose config cannot be read,
 * because it is gated or private or was never published, is recorded with a
 * null config key rather than dropped, so the row still appears on the page
 * and says that it could not be sized.
 */
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const OUTPUT_PATH = join(projectRoot, 'src', 'lib', 'reachability', 'dataset.generated.ts')

const LEADERBOARD_URL = 'https://clef-evals.workers-ai-mle.workers.dev/data/leaderboard.json'
const HF_CONFIG_BASE = 'https://huggingface.co'
const CONCURRENCY = 6
const REQUEST_TIMEOUT_MS = 30000

/** Fetches a URL as JSON, or returns null on any non success answer. */
async function fetchJson(url) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { accept: 'application/json' },
    })
    if (!response.ok) return null
    return await response.json()
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

/**
 * The HuggingFace repository a model's config lives in.
 *
 * A base checkpoint is the natural answer when the leaderboard names one. When
 * it does not, the model card URL is the next best thing, because it points at
 * the repository the checkpoint was published to. Anything that is not a
 * HuggingFace repository, which is every closed hosted API, resolves to null.
 */
function resolveRepo(model) {
  if (typeof model.base_model === 'string' && model.base_model.trim() !== '') {
    return model.base_model.trim()
  }
  const url = typeof model.model_url === 'string' ? model.model_url : ''
  const marker = 'huggingface.co/'
  const at = url.indexOf(marker)
  if (at === -1) return null
  const rest = url.slice(at + marker.length)
  const segments = rest.split(/[/?#]/).filter(Boolean)
  if (segments.length < 2) return null
  return `${segments[0]}/${segments[1]}`
}

/**
 * The dtype or quantisation a config declares, for display only.
 *
 * The calculator runs its own format detector over the whole config, so this
 * string never decides a size. It is here so a row whose config could not be
 * read can still name the format the model card claims.
 */
function formatHintFrom(config) {
  if (!config || typeof config !== 'object') return null
  const quant = config.quantization_config
  const nested = config.text_config && typeof config.text_config === 'object' ? config.text_config : {}
  if (quant && typeof quant === 'object') {
    for (const key of ['store_dtype', 'quant_method', 'fmt']) {
      const value = quant[key]
      if (typeof value === 'string' && value.trim() !== '') return value
    }
  }
  for (const source of [config, nested]) {
    for (const key of ['torch_dtype', 'dtype']) {
      const value = source[key]
      if (typeof value === 'string' && value.trim() !== '') return value
    }
  }
  return null
}

/** Reads a finite number, or null when the value is absent or not numeric. */
function numberOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** Reads a string, or null when the value is absent or not a string. */
function stringOrNull(value) {
  return typeof value === 'string' && value !== '' ? value : null
}

/**
 * Fetches the config for every repository, with a bounded pool so a wide model
 * list does not open one connection for each row at once.
 */
async function fetchConfigs(repos) {
  const configs = new Map()
  const queue = [...repos]
  async function worker() {
    for (;;) {
      const repo = queue.shift()
      if (repo === undefined) return
      const config = await fetchJson(`${HF_CONFIG_BASE}/${repo}/raw/main/config.json`)
      if (config && typeof config === 'object' && !Array.isArray(config)) {
        configs.set(repo, config)
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, worker))
  return configs
}

/** Serialises a value as a TypeScript literal, with a stable key order. */
function literal(value) {
  return JSON.stringify(value, null, 2)
}

async function main() {
  const board = await fetchJson(LEADERBOARD_URL)
  if (!board || !Array.isArray(board.models)) {
    throw new Error(`The leaderboard at ${LEADERBOARD_URL} could not be read.`)
  }

  const repos = new Set()
  for (const model of board.models) {
    const repo = resolveRepo(model)
    if (repo) repos.add(repo)
  }

  console.log(`Read ${board.models.length} models. Fetching ${repos.size} configs.`)
  const configs = await fetchConfigs([...repos].sort())
  console.log(`Read ${configs.size} of ${repos.size} configs.`)

  const models = board.models.map((model) => {
    const repo = resolveRepo(model)
    const config = repo ? configs.get(repo) : undefined
    const latency = model.latency && typeof model.latency === 'object' ? model.latency : {}
    return {
      engine: String(model.engine),
      name: String(model.name ?? model.engine),
      org: stringOrNull(model.org),
      kind: String(model.kind ?? 'unknown'),
      baseModel: stringOrNull(model.base_model),
      modelUrl: stringOrNull(model.model_url),
      params: numberOrNull(model.params),
      index: numberOrNull(model.index),
      latencyMedian: numberOrNull(latency.median),
      latencyP95: numberOrNull(latency.p95),
      closed: model.closed === true,
      source: stringOrNull(model.source),
      panelCoverage: numberOrNull(model.panel_coverage),
      missing: Array.isArray(model.missing)
        ? model.missing.filter((id) => typeof id === 'number')
        : [],
      formatHint: formatHintFrom(config),
      configKey: config ? repo : null,
    }
  })

  // The map is sorted by key so a rerun that reads the same configs writes the
  // same file. Models keep the leaderboard order, which is its own ranking.
  const configEntries = [...configs.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))

  const areas = (Array.isArray(board.areas) ? board.areas : []).map((area) => ({
    id: String(area.id),
    label: String(area.label),
    weight: numberOrNull(area.weight) ?? 0,
  }))

  const upstream = board.upstream && typeof board.upstream === 'object' ? board.upstream : {}

  const source = {
    generatedUtc: new Date().toISOString(),
    leaderboardUrl: LEADERBOARD_URL,
    upstreamLabel: String(upstream.label ?? 'Decision Index'),
    upstreamGeneratedUtc: String(upstream.generated_utc ?? board.generated_utc ?? ''),
    upstreamHardware: String(upstream.hardware ?? 'unknown'),
    upstreamUrl: String(upstream.url ?? ''),
    panelSize: numberOrNull(board.panel_size) ?? 0,
    areas,
  }

  const lines = []
  lines.push('/**')
  lines.push(' * Generated by scripts/fetch-reachability.mjs. Do not edit by hand.')
  lines.push(' *')
  lines.push(' * The model list and the scores come from the clef-evals Decision Model')
  lines.push(' * Leaderboard. The configs come from each model base checkpoint on')
  lines.push(' * HuggingFace. Rerun `pnpm data:reachability` to refresh both.')
  lines.push(' */')
  lines.push("import type { RawConfig } from '../model-config'")
  lines.push('')
  lines.push("import type { ReachabilityModelRecord, ReachabilitySource } from './types'")
  lines.push('')
  lines.push(`export const REACHABILITY_SOURCE: ReachabilitySource = ${literal(source)}`)
  lines.push('')
  lines.push(
    `export const REACHABILITY_MODELS: ReachabilityModelRecord[] = ${literal(models)}`,
  )
  lines.push('')
  lines.push(`export const REACHABILITY_CONFIGS: Record<string, RawConfig> = ${literal(Object.fromEntries(configEntries))} as Record<string, RawConfig>`)
  lines.push('')

  await mkdir(dirname(OUTPUT_PATH), { recursive: true })
  await writeFile(OUTPUT_PATH, lines.join('\n'), 'utf8')

  const sizeable = models.filter((model) => model.configKey !== null).length
  console.log(`Wrote ${OUTPUT_PATH}.`)
  console.log(`${models.length} models, ${sizeable} with a config, ${configEntries.length} configs.`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
