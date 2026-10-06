import { REACHABILITY_SORTS, type ReachabilitySortKey } from '@/lib/reachability'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface ReachabilitySortPickerProps {
  value: ReachabilitySortKey
  onChange: (value: ReachabilitySortKey) => void
}

/**
 * The order the board is read in.
 *
 * It sits beside the table rather than in the workload form, because it changes
 * the view and not the answer, and the reader who lands on rank 1 with a low
 * Decision Index can see from there that the board is ranked by hardware.
 */
export default function ReachabilitySortPicker({
  value,
  onChange,
}: ReachabilitySortPickerProps) {
  const selected = REACHABILITY_SORTS.find((sort) => sort.id === value)

  return (
    <div className="flex w-full flex-col gap-1 sm:w-64">
      <label htmlFor="reachability-sort" className="text-xs font-medium">
        Order
      </label>
      <Select
        value={value}
        onValueChange={(next) => {
          const option = REACHABILITY_SORTS.find((sort) => sort.id === next)
          if (option) onChange(option.id)
        }}
      >
        <SelectTrigger id="reachability-sort" className="w-full">
          <SelectValue>{selected?.label ?? 'Reachability'}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {REACHABILITY_SORTS.map((sort) => (
            <SelectItem key={sort.id} value={sort.id}>
              {sort.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {selected?.hint ?? REACHABILITY_SORTS[0].hint}
      </p>
    </div>
  )
}
