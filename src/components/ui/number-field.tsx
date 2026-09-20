import { Input } from '@/components/ui/input'

interface NumberFieldProps {
  id: string
  label: string
  hint?: string
  value: string
  onChange: (value: string) => void
  /** Marks the field when the estimator rejected this input. */
  invalid?: boolean
  min?: number
  /** A fraction is allowed, which changes the step and the on-screen keyboard. */
  decimal?: boolean
}

/**
 * A labelled numeric input with an optional hint.
 *
 * Both calculators are mostly made of these, and the label, step, input mode and
 * invalid marking have to agree across all of them, so the field is defined once.
 */
export default function NumberField({
  id,
  label,
  hint,
  value,
  onChange,
  invalid,
  min,
  decimal,
}: NumberFieldProps) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <Input
        id={id}
        type="number"
        inputMode={decimal ? 'decimal' : 'numeric'}
        step={decimal ? 'any' : 1}
        min={min}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-invalid={invalid}
      />
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
