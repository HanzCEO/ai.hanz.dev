import { WEIGHT_DTYPES, type WeightDtype } from '@/lib/reap'
import type { GpuSpec } from '@/lib/hardware'
import type { SupportLevel } from '@/lib/support'

import SupportTag from '@/components/ui/support-tag'
import Marquee from '@/components/ui/marquee'
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
      reason: 'This is the precision of most published checkpoints. It runs on every GPU in the list.',
    }
  }

  if (dtype === 'FP8') {
    if (gpu.fp8DenseTflops === null) {
      return {
        level: 'unsupported',
        reason: `${gpu.label} has no FP8 tensor path. The calibration therefore runs at the BF16 throughput.`,
      }
    }
    return {
      level: 'supported',
      reason: `This halves the resident expert block. It runs at ${gpu.fp8DenseTflops.toLocaleString('en-US')} TFLOPS dense on ${gpu.label}.`,
    }
  }

  return {
    level: 'untested',
    reason: 'It needs a checkpoint that is already quantised to 4 bits. No published recipe calibrates one.',
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
                  <Marquee>{spec.label}</Marquee>
                  <SupportTag level={tag.level} className="shrink-0" />
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
