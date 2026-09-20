/**
 * Number and duration formatting shared by every calculator.
 *
 * These live here rather than inside a single tool's folder because a figure
 * like a byte count or a run duration is presented the same way whichever
 * calculator produced it. A second copy would be a second chance for the two
 * pages to disagree about how 38 TB reads.
 */

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

/**
 * Reads a duration the way a person would say it. An estimate spans seconds to
 * days, so the unit has to follow the magnitude.
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'no time'
  // A positive duration below a second is real work, so it must not round down
  // to "0 seconds" and read as no work at all.
  if (seconds < 1) return 'less than a second'
  if (seconds < 90) {
    const whole = Math.round(seconds)
    // The unit has to agree with the value, so a rounded 1 is singular.
    return `${whole} ${whole === 1 ? 'second' : 'seconds'}`
  }
  const minutes = seconds / 60
  if (minutes < 90) return `${minutes < 10 ? minutes.toFixed(1) : Math.round(minutes)} minutes`
  const hours = seconds / 3600
  if (hours < 72) return `${hours.toFixed(1)} hours`
  return `${(hours / 24).toFixed(1)} days`
}
