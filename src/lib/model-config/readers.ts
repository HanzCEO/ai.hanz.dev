import type { RawConfig } from './types'

/**
 * Config field readers. Configs in the wild omit fields freely, and the same
 * value is sometimes a number and sometimes a numeric string, so everything is
 * read defensively and absence is handled rather than assumed away.
 */

/** Keys under which multimodal configs nest their language model fields. */
const NESTED_KEYS = ['text_config', 'llm_config', 'language_config'] as const

export function readNumber(config: RawConfig, key: string): number | undefined {
  const value = config[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
    return Number(value)
  }
  return undefined
}

export function readNumberArray(config: RawConfig, key: string): number[] | undefined {
  const value = config[key]
  if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'number')) {
    return value as number[]
  }
  return undefined
}

export function readStringArray(config: RawConfig, key: string): string[] | undefined {
  const value = config[key]
  if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'string')) {
    return value as string[]
  }
  return undefined
}

/**
 * Reads a per layer flag list as booleans.
 *
 * Releases describe the same pattern in two ways: an array of booleans, and an
 * array of zeros and ones. MiMo-V2.6 writes moe_layer_freq as [0, 1, 1, ...],
 * and hybrid_layer_pattern as [0, 1, 1, ...] where 0 marks a global attention
 * layer and 1 marks a sliding window one. Both forms mean the same thing here,
 * so a numeric entry is read as Boolean(value). A mixed array is rejected
 * rather than guessed at, and an empty array carries no pattern at all.
 */
export function readFlagArray(config: RawConfig, key: string): boolean[] | undefined {
  const value = config[key]
  if (!Array.isArray(value) || value.length === 0) return undefined
  if (value.every((item) => typeof item === 'boolean')) return value as boolean[]
  if (value.every((item) => typeof item === 'number' && Number.isFinite(item))) {
    return (value as number[]).map((item) => item !== 0)
  }
  return undefined
}

export function readString(config: RawConfig, key: string): string | undefined {
  const value = config[key]
  return typeof value === 'string' ? value : undefined
}

export function readBoolean(config: RawConfig, key: string): boolean | undefined {
  const value = config[key]
  return typeof value === 'boolean' ? value : undefined
}

export function readObject(config: RawConfig, key: string): RawConfig | undefined {
  const value = config[key]
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as RawConfig
  return undefined
}

/**
 * Reads the first key that is present, in the order given. MoE configs name the
 * routed expert count differently across families, so callers pass candidates.
 */
export function readNumberFrom(config: RawConfig, keys: readonly string[]): number | undefined {
  for (const key of keys) {
    const value = readNumber(config, key)
    if (value !== undefined) return value
  }
  return undefined
}

export function readStringFrom(config: RawConfig, keys: readonly string[]): string | undefined {
  for (const key of keys) {
    const value = readString(config, key)
    if (value !== undefined) return value
  }
  return undefined
}

export function readArrayFrom(
  config: RawConfig,
  keys: readonly string[],
  kind: 'number' | 'string',
): number[] | string[] | undefined {
  for (const key of keys) {
    const value = kind === 'number' ? readNumberArray(config, key) : readStringArray(config, key)
    if (value !== undefined) return value
  }
  return undefined
}

/**
 * Multimodal configs (DeepSeek-V4.1-Flash, Qwen3.5, Gemma 4) keep the language
 * model fields under a nested key. Returns the inner config plus the outer one,
 * since a few fields such as quantization_config only live outside.
 */
export function unwrapConfig(raw: RawConfig): { inner: RawConfig; outer: RawConfig } {
  for (const key of NESTED_KEYS) {
    const nested = raw[key]
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      return { inner: nested as RawConfig, outer: raw }
    }
  }
  return { inner: raw, outer: raw }
}

/** Converts a torch dtype name into bytes per element. */
export function dtypeNameToBytes(name: string | undefined): number | undefined {
  if (!name) return undefined
  const normalized = name.toLowerCase()
  if (normalized.includes('float32') || normalized === 'fp32') return 4
  if (normalized.includes('float16') || normalized === 'fp16' || normalized === 'half') return 2
  if (normalized.includes('bfloat16') || normalized === 'bf16') return 2
  if (normalized.includes('float8') || normalized === 'fp8') return 1
  if (normalized.includes('int8')) return 1
  // A 4 bit element is half a byte. The block scale is added by the size
  // formula that actually packs the cache, so it is not counted here.
  if (
    normalized.includes('fp4') ||
    normalized.includes('mxfp4') ||
    normalized.includes('nvfp4') ||
    normalized.includes('int4') ||
    normalized.includes('4bit')
  ) {
    return 0.5
  }
  return undefined
}
