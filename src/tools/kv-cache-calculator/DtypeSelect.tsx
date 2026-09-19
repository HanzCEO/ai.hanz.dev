import {
  DTYPES,
  dtypeLabel,
  isDtypeId,
  type DtypeId,
  type DtypeSupport,
} from '@/lib/kvcache'

import SupportTag from '@/components/ui/support-tag'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface DtypeSelectProps {
  id: string
  label: string
  value: DtypeId
  onValueChange: (value: DtypeId) => void
  support: DtypeSupport[]
  disabled?: boolean
}

/**
 * Every dtype is always offered. The tag says whether the architecture actually
 * supports it, and the selected option explains why.
 */
export default function DtypeSelect({
  id,
  label,
  value,
  onValueChange,
  support,
  disabled,
}: DtypeSelectProps) {
  const supportByDtype = new Map(support.map((entry) => [entry.dtype, entry]))
  const selected = supportByDtype.get(value)

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <Select
        value={value}
        onValueChange={(next) => {
          // Radix mirrors the value into a hidden native select, which reports an
          // empty string when the controlled value changes while the list is
          // closed. Ignore anything that is not a real dtype.
          if (isDtypeId(next)) onValueChange(next)
        }}
        disabled={disabled}
      >
        <SelectTrigger id={id} className="w-full">
          {/* Rendered explicitly so the trigger shows a label even when the
              value was set programmatically and the items are not mounted. */}
          <SelectValue>{dtypeLabel(value)}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {DTYPES.map((dtype) => {
            const entry = supportByDtype.get(dtype.id)
            return (
              <SelectItem key={dtype.id} value={dtype.id}>
                <span className="flex w-full items-center justify-between gap-4">
                  <span>{dtype.label}</span>
                  {entry && <SupportTag level={entry.level} />}
                </span>
              </SelectItem>
            )
          })}
        </SelectContent>
      </Select>
      {selected && (
        <p className="text-xs text-muted-foreground">
          <SupportTag level={selected.level} className="mr-1.5" />
          {selected.reason}
        </p>
      )}
    </div>
  )
}
