import { WEIGHT_DTYPES, type WeightDtype } from '@/lib/reap'
import type { GpuSpec } from '@/lib/hardware'
import type { SupportLevel } from '@/lib/support'
import { isFourBitFormat, isFp8Format, weightFormatLabel } from '@/lib/weight-format'

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
 * Grounds each format in what the card can actually do.
 *
 * FP8 needs an FP8 tensor path, which is Blackwell, Hopper, Ada, CDNA 3, and
 * RDNA 4. A 4 bit format needs an FP4 tensor path, which is Blackwell and
 * RDNA 4. Where the card has no FP4 path, the block is still a quarter of its
 * BF16 size, but the calibration dequantizes it and runs at the FP8 or BF16
 * rate.
 */
function tagFor(dtype: WeightDtype, gpu: GpuSpec): Tag {
  if (dtype === 'BF16' || dtype === 'FP16') {
    return {
      level: 'supported',
      reason: 'This is the format of most published checkpoints. It runs on every GPU in the list.',
    }
  }

  if (isFp8Format(dtype)) {
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

  if (isFourBitFormat(dtype)) {
    if (gpu.fp4DenseTflops === null) {
      return {
        level: 'untested',
        reason: `${gpu.label} has no FP4 tensor path, so the calibration dequantizes the block and runs at the FP8 or BF16 rate.`,
      }
    }
    return {
      level: 'supported',
      reason: `This cuts the resident expert block to about a quarter. It runs at ${gpu.fp4DenseTflops.toLocaleString('en-US')} TFLOPS dense on ${gpu.label}.`,
    }
  }

  return {
    level: 'untested',
    reason: 'It needs a checkpoint that is already quantised to this format.',
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
        Weight format
      </label>
      <Select
        value={value}
        onValueChange={(next) => {
          if (WEIGHT_DTYPES.some((spec) => spec.id === next)) onValueChange(next as WeightDtype)
        }}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue>{weightFormatLabel(value)}</SelectValue>
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
