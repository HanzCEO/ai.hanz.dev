/**
 * Reads a duration the way a person would say it. A REAP estimate spans
 * seconds to days, so the unit has to follow the magnitude.
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'no time'
  if (seconds < 90) return `${Math.round(seconds)} seconds`
  const minutes = seconds / 60
  if (minutes < 90) return `${minutes < 10 ? minutes.toFixed(1) : Math.round(minutes)} minutes`
  const hours = seconds / 3600
  if (hours < 72) return `${hours.toFixed(1)} hours`
  return `${(hours / 24).toFixed(1)} days`
}
