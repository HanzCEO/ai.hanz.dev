import type { WorkloadMix } from './types'

/**
 * The window the chart is drawn over.
 *
 * A million tokens is the largest context any current model serves, so it is
 * the point where the decision has the most money behind it.
 */
export const CONTEXT_WINDOW_TOKENS = 1_000_000

/** A summary keeps at least this share of the session. */
export const MIN_COMPRESSION_PERCENT = 1

/** Keeping everything is the same as not summarising at all. */
export const MAX_COMPRESSION_PERCENT = 100

export const DEFAULT_COMPRESSION_PERCENT = 30

export interface WorkloadPreset {
  id: string
  label: string
  mix: WorkloadMix
  note: string
}

/**
 * The three ways people actually use a long session.
 *
 * The cached share is what decides the answer. A coding session resends the
 * same repository and the same instructions on every turn, so almost all of its
 * input is cached and the cache is cheap. A bug hunt rewrites the failing case
 * often, so more of its input misses.
 */
export const WORKLOAD_PRESETS: WorkloadPreset[] = [
  {
    id: 'coding',
    label: 'Normal coding',
    mix: { missPercent: 4.5, cachePercent: 95, outputPercent: 0.5 },
    note: 'Most turns resend the same files and instructions, so nearly every input token hits the cache.',
  },
  {
    id: 'bug-hunting',
    label: 'Bug hunting',
    mix: { missPercent: 15, cachePercent: 80, outputPercent: 5 },
    note: 'You paste new logs and new failures each turn, so more input misses the cache.',
  },
  {
    id: 'assistant',
    label: 'Personal assistant',
    mix: { missPercent: 10, cachePercent: 89, outputPercent: 1 },
    note: 'A long conversation with new questions and short answers.',
  },
]

export const DEFAULT_WORKLOAD_PRESET = 'coding'

export function findWorkloadPreset(id: string): WorkloadPreset | undefined {
  return WORKLOAD_PRESETS.find((preset) => preset.id === id)
}

export interface PricePreset {
  id: string
  label: string
  /** Dollars for 1M input tokens that miss the cache. */
  inputPrice: number
  /** Dollars for 1M input tokens that hit the cache. */
  cachedInputPrice: number
  /** Dollars for 1M output tokens. */
  outputPrice: number
  note: string
}

/**
 * List rates for one model from each major provider, in dollars for 1M tokens.
 *
 * These are published list prices at the time of writing, and they change. The
 * three price fields are editable, so a stale entry here costs the reader
 * nothing but a correction.
 */
export const PRICE_PRESETS: PricePreset[] = [
  {
    id: 'anthropic-sonnet',
    label: 'Anthropic Sonnet tier',
    inputPrice: 3,
    cachedInputPrice: 0.3,
    outputPrice: 15,
    note: 'List rates at the time of writing, and every price field is editable.',
  },
  {
    id: 'gpt-workhorse',
    label: 'GPT workhorse tier',
    inputPrice: 2,
    cachedInputPrice: 0.2,
    outputPrice: 12,
    note: 'List rates at the time of writing, and every price field is editable.',
  },
  {
    id: 'qwen-workhorse',
    label: 'Qwen workhorse tier',
    inputPrice: 2,
    cachedInputPrice: 0.2,
    outputPrice: 6,
    note: 'List rates at the time of writing, and every price field is editable.',
  },
  {
    id: 'glm-flagship',
    label: 'GLM flagship',
    inputPrice: 1.4,
    cachedInputPrice: 0.26,
    outputPrice: 4.4,
    note: 'List rates at the time of writing, and every price field is editable.',
  },
  {
    id: 'deepseek-flash',
    label: 'DeepSeek Flash',
    inputPrice: 0.14,
    cachedInputPrice: 0.0028,
    outputPrice: 0.28,
    note: 'List rates at the time of writing, and every price field is editable.',
  },
]

export const DEFAULT_PRICE_PRESET_ID = 'anthropic-sonnet'

export function findPricePreset(id: string): PricePreset | undefined {
  return PRICE_PRESETS.find((preset) => preset.id === id)
}
