const BINARY_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB'] as const

export interface FormattedBytes {
  /** Scaled magnitude, for example 4.5. */
  value: number
  /** Binary unit matching the magnitude. */
  unit: string
  /** Value and unit together, for example "4.50 GiB". */
  text: string
  /** Exact byte count with thousands separators. */
  exact: string
}

/** Formats a byte count with binary units, scaling to whichever unit fits. */
export function formatBytes(bytes: number): FormattedBytes {
  const safe = Number.isFinite(bytes) && bytes > 0 ? bytes : 0

  let index = 0
  let value = safe
  while (value >= 1024 && index < BINARY_UNITS.length - 1) {
    value /= 1024
    index += 1
  }

  // Bytes are exact, larger units read better with two decimals until they get big.
  const decimals = index === 0 ? 0 : value >= 100 ? 1 : 2

  return {
    value,
    unit: BINARY_UNITS[index],
    text: `${value.toFixed(decimals)} ${BINARY_UNITS[index]}`,
    exact: formatExact(safe),
  }
}

export function formatExact(bytes: number): string {
  if (!Number.isFinite(bytes)) return '0'
  return Math.round(bytes).toLocaleString('en-US')
}

/** Formats a token count, for example 131,072 or 1.05M. */
export function formatTokens(tokens: number): string {
  if (!Number.isFinite(tokens)) return '0'
  if (tokens >= 1_000_000 && tokens % 1_000_000 === 0) return `${tokens / 1_000_000}M`
  return tokens.toLocaleString('en-US')
}
