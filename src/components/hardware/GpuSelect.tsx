import { GPU_PRESETS, GPU_VENDOR_LABELS, GPU_VENDORS, findGpu } from '@/lib/hardware'

import Marquee from '@/components/ui/marquee'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface GpuSelectProps {
  id: string
  value: string
  onValueChange: (value: string) => void
}

/**
 * Groups every card by vendor, so the datacenter parts and the consumer parts
 * do not run together in one flat list of forty entries.
 */
export default function GpuSelect({ id, value, onValueChange }: GpuSelectProps) {
  const selected = findGpu(value)

  return (
    <Select
      value={value}
      onValueChange={(next) => {
        // Radix mirrors the value into a hidden native select, which reports an
        // empty string while the list is closed. Ignore anything unknown.
        if (findGpu(next)) onValueChange(next)
      }}
    >
      <SelectTrigger id={id} className="w-full">
        <SelectValue>
          {selected ? `${selected.label} · ${selected.vramGiB} GB` : 'Pick a card'}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {GPU_VENDORS.map((vendor) => (
          <SelectGroup key={vendor}>
            <SelectLabel>{GPU_VENDOR_LABELS[vendor]}</SelectLabel>
            {GPU_PRESETS.filter((gpu) => gpu.vendor === vendor).map((gpu) => (
              <SelectItem key={gpu.id} value={gpu.id}>
                <span className="flex w-full items-center justify-between gap-4">
                  <Marquee>{gpu.label}</Marquee>
                  <span className="text-muted-foreground shrink-0 text-xs tabular-nums">
                    {gpu.vramGiB} GB
                  </span>
                </span>
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  )
}
