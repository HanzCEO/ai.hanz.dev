import { formatBytes, formatExact } from '@/lib/format'

import type { SizeLine, TokenSizeResult } from './types'

/**
 * Converts a token count into bytes and then into readable sizes.
 *
 * The bytes come from the tokenizer profile's bytes per token, and the units
 * are the decimal set people actually name — KB, MB, GB, TB, PB — where one
 * GB is 1,000,000,000 bytes. The binary figure is given alongside the headline
 * because the rest of this site sizes memory in GiB, and "how many GB" is
 * asked at both scales.
 */

const DECIMAL_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'] as const
export type DecimalUnit = (typeof DECIMAL_UNITS)[number]

const SUFFIX_MULTIPLIERS: Record<string, number> = {
  k: 1_000,
  m: 1_000_000,
  b: 1_000_000_000,
  t: 1_000_000_000_000,
}

/**
 * Reads a token amount the way a person types one: plain digits with optional
 * thousands separators, or a number with a k, m, b, or t suffix such as 128k,
 * 1.5M, or 1T. Returns null for anything that is not a positive amount.
 */
export function parseTokenAmount(text: string): number | null {
  const trimmed = text.trim()
  if (trimmed === '') return null

  const withSuffix = /^(\d+(?:\.\d+)?)\s*([kmbt])$/i.exec(trimmed)
  if (withSuffix) {
    const base = Number(withSuffix[1])
    const multiplier = SUFFIX_MULTIPLIERS[withSuffix[2].toLowerCase()]
    const value = base * multiplier
    return Number.isFinite(value) && value >= 1 ? value : null
  }

  const digits = trimmed.replace(/,/g, '')
  if (!/^\d+$/.test(digits)) return null
  const value = Number(digits)
  return Number.isSafeInteger(value) && value >= 1 ? value : null
}

/** Formats one converted unit value with a readable number of decimals. */
function formatUnitValue(value: number): string {
  if (value >= 1_000_000) return value.toLocaleString('en-US', { maximumFractionDigits: 1 })
  if (value >= 100) return Math.round(value).toLocaleString('en-US')
  if (value >= 1) return String(+value.toFixed(2))
  return value.toPrecision(3)
}

/** The same byte count in every decimal unit, from bytes to petabytes. */
export function bytesInDecimalUnits(bytes: number): SizeLine[] {
  const lines: SizeLine[] = []
  for (let i = 0; i < DECIMAL_UNITS.length; i++) {
    const unit = DECIMAL_UNITS[i]
    const value = bytes / 1000 ** i
    lines.push({ unit, value, text: `${formatUnitValue(value)} ${unit}` })
  }
  return lines
}

/**
 * The unit whose magnitude reads best on a card: the largest unit the value
 * still fills past one, falling back to bytes.
 */
export function headlineUnit(bytes: number): DecimalUnit {
  const lines = bytesInDecimalUnits(bytes)
  let pick: DecimalUnit = lines[lines.length - 1].unit
  for (const line of lines) {
    if (line.value >= 1) pick = line.unit
  }
  return pick
}

/**
 * Builds the answer: bytes, the headline in the unit that fits, the decimal
 * table, the binary equivalent, and the exact byte count.
 */
export function estimateTokenSize(tokens: number, bytesPerToken: number): TokenSizeResult {
  const bytes = tokens * bytesPerToken
  const unit = headlineUnit(bytes)
  const lines = bytesInDecimalUnits(bytes)
  const scaled = lines.find((entry) => entry.unit === unit)

  const binary = formatBytes(bytes)
  return {
    tokens,
    bytes,
    bytesPerToken,
    headline: scaled ? `${formatUnitValue(scaled.value)} ${unit}` : '',
    lines,
    binary: binary.text,
    exactBytes: formatExact(bytes),
  }
}

/** Formats a token count for short labels, for example "1B" or "128K". */
export function formatTokenAmountShort(tokens: number): string {
  const short = (value: number, suffix: string): string => {
    const rounded = +value.toFixed(1)
    const text = Number.isInteger(rounded) ? String(Math.round(rounded)) : rounded.toFixed(1)
    return `${text}${suffix}`
  }
  if (tokens >= 1_000_000_000_000) return short(tokens / 1_000_000_000_000, 'T')
  if (tokens >= 1_000_000_000) return short(tokens / 1_000_000_000, 'B')
  if (tokens >= 1_000_000) return short(tokens / 1_000_000, 'M')
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`
  return String(Math.round(tokens))
}