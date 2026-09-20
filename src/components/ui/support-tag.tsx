import type { SupportLevel } from '@/lib/support'

const LEVEL_STYLES: Record<SupportLevel, string> = {
  supported: 'text-emerald-600 dark:text-emerald-400',
  untested: 'text-amber-600 dark:text-amber-400',
  unsupported: 'text-rose-600 dark:text-rose-400',
}

const LEVEL_LABELS: Record<SupportLevel, string> = {
  supported: 'supported',
  untested: 'untested',
  unsupported: 'unsupported',
}

/**
 * Tags an option with how well it is grounded, so a dropdown never asserts
 * support without saying why. Shared by the KV cache dtype picker and the REAP
 * precision picker.
 */
export default function SupportTag({
  level,
  className,
}: {
  level: SupportLevel
  className?: string
}) {
  return (
    <span className={`text-[0.7rem] ${LEVEL_STYLES[level]} ${className ?? ''}`}>
      {LEVEL_LABELS[level]}
    </span>
  )
}
