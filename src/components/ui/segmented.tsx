import type { ReactNode } from 'react'

/**
 * A row of mutually exclusive buttons, styled as one bordered group.
 *
 * Both calculators present a small set of choices this way, the provider and
 * the model source, and the markup is identical apart from the labels. Sharing
 * it means a change to the active state or the pressed attribute lands on every
 * picker at once.
 */
export interface SegmentedOption<T extends string> {
  id: T
  label: string
  /** Shown under the group while this option is selected. */
  hint?: string
}

interface SegmentedControlProps<T extends string> {
  legend: string
  options: readonly SegmentedOption<T>[]
  value: T
  onValueChange: (value: T) => void
  /** Wraps the row onto more than one line instead of letting it run wide. */
  wrap?: boolean
  /** Overrides the hint line, which otherwise follows the selected option. */
  hint?: ReactNode
}

export default function SegmentedControl<T extends string>({
  legend,
  options,
  value,
  onValueChange,
  wrap = false,
  hint,
}: SegmentedControlProps<T>) {
  const selectedHint = options.find((option) => option.id === value)?.hint

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-2 text-sm font-medium">{legend}</legend>
      <div
        className={`border-border inline-flex w-fit${wrap ? ' flex-wrap' : ''} rounded-lg border p-0.5`}
      >
        {options.map((option) => {
          const active = value === option.id
          return (
            <button
              key={option.id}
              type="button"
              aria-pressed={active}
              onClick={() => onValueChange(option.id)}
              className={`rounded-md px-3 py-1 text-sm transition-colors ${
                active
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {option.label}
            </button>
          )
        })}
      </div>
      {(hint ?? selectedHint) && (
        <p className="text-xs text-muted-foreground">{hint ?? selectedHint}</p>
      )}
    </fieldset>
  )
}
