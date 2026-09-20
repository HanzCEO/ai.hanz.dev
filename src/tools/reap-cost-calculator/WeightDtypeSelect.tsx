import { WEIGHT_DTYPES, type WeightDtype } from '@/lib/reap'
import type { GpuSpec } from '@/lib/hardware'
import type { SupportLevel } from '@/lib/support'

import SupportTag from '@/components/ui/support-tag'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface Tag {
  level: SupportLevel
  reason: string
}

/**
 * Grounds each precision in what the card can actually do. FP8 is only offered
 * as supported where the architecture has a real FP8 tensor path, which is
 * Blackwell, Hopper, CDNA 3, and RDNA 4.
 */
function tagFor(dtype: WeightDtype, gpu: GpuSpec): Tag {
  if (dtype === 'BF16') {
    return {
      level: 'supported',
      reason: 'Reads at the precision most checkpoints ship in, and runs on every card listed.',
    }
  }

  if (dtype === 'FP8') {
    if (gpu.fp8DenseTflops === null) {
      return {
        level: 'unsupported',
        reason: `${gpu.label} has no FP8 tensor path, so the pass would run at the BF16 rate.`,
      }
    }
    return {
      level: 'supported',
      reason: `Halves the resident block and runs at ${gpu.fp8DenseTflops.toLocaleString('en-US')} TFLOPS dense on ${gpu.label}.`,
    }
  }

  return {
    level: 'untested',
    reason: 'Needs a checkpoint that was already quantised to four bits. Calibrating one is not a published recipe.',
  }
}

interface WeightDtypeSelectProps {
  id: string
  value: WeightDtype
  gpu: GpuSpec
  onValueChange: (value: WeightDtype) => void
}

export default function WeightDtypeSelect({
  id,
  value,
  gpu,
  onValueChange,
}: WeightDtypeSelectProps) {
  const selected = tagFor(value, gpu)

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        Weight precision
      </label>
      <Select
        value={value}
        onValueChange={(next) => {
          if (WEIGHT_DTYPES.some((spec) => spec.id === next)) onValueChange(next as WeightDtype)
        }}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue>{value}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {WEIGHT_DTYPES.map((spec) => {
            const tag = tagFor(spec.id, gpu)
            return (
              <SelectItem key={spec.id} value={spec.id}>
                <span className="flex w-full items-center justify-between gap-4">
                  <span>{spec.label}</span>
                  <SupportTag level={tag.level} />
                </span>
              </SelectItem>
            )
          })}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        <SupportTag level={selected.level} className="mr-1.5" />
        {selected.reason}
      </p>
    </div>
  )
}
