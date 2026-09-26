import {
  PRICE_PRESETS,
  WORKLOAD_PRESETS,
  findPricePreset,
  findWorkloadPreset,
  type CompressionInputField,
} from '@/lib/compression'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import Marquee from '@/components/ui/marquee'
import NumberField from '@/components/ui/number-field'
import SegmentedControl from '@/components/ui/segmented'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

import type { CompressionFormInputs } from './useCompressionState'

/** The select entry that stands for prices typed by hand. */
const CUSTOM_PRICE_ID = 'custom'

/** The segmented entry that stands for a mix typed by hand. */
const CUSTOM_MIX_ID = 'custom'

/** True when a text field holds the same number as a preset value. */
function sameNumber(field: string, value: number): boolean {
  return Number(field) === value
}

/** The price preset the current fields describe, or the custom entry. */
function activePricePresetId(inputs: CompressionFormInputs): string {
  const match = PRICE_PRESETS.find(
    (preset) =>
      sameNumber(inputs.inputPrice, preset.inputPrice) &&
      sameNumber(inputs.cachedInputPrice, preset.cachedInputPrice) &&
      sameNumber(inputs.outputPrice, preset.outputPrice),
  )
  return match?.id ?? CUSTOM_PRICE_ID
}

/** The workload preset the current fields describe, or the custom entry. */
function activeWorkloadPresetId(inputs: CompressionFormInputs): string {
  const match = WORKLOAD_PRESETS.find(
    (preset) =>
      sameNumber(inputs.missPercent, preset.mix.missPercent) &&
      sameNumber(inputs.cachePercent, preset.mix.cachePercent) &&
      sameNumber(inputs.outputPercent, preset.mix.outputPercent),
  )
  return match?.id ?? CUSTOM_MIX_ID
}

const MIX_OPTIONS = [
  ...WORKLOAD_PRESETS.map((preset) => ({
    id: preset.id,
    label: preset.label,
    hint: preset.note,
  })),
  { id: CUSTOM_MIX_ID, label: 'Custom', hint: 'Your own token mix.' },
]

interface CompressionFormProps {
  inputs: CompressionFormInputs
  update: (patch: Partial<CompressionFormInputs>) => void
  /** The input the estimator rejected, so it can be marked in the form. */
  invalidField: CompressionInputField | null
  /** The message the estimator returned, shown above the fields. */
  computeError: string | null
}

export default function CompressionForm({
  inputs,
  update,
  invalidField,
  computeError,
}: CompressionFormProps) {
  const priceId = activePricePresetId(inputs)
  const pricePreset = findPricePreset(priceId)
  const mixId = activeWorkloadPresetId(inputs)
  const compressionValue = Number(inputs.compressionPercent)
  const sliderValue = Number.isFinite(compressionValue) ? compressionValue : 30

  return (
    <form className="flex flex-col gap-8" onSubmit={(event) => event.preventDefault()}>
      {computeError && (
        <Alert variant="destructive">
          <AlertTitle>Those numbers do not describe a session</AlertTitle>
          <AlertDescription>
            <p>{computeError}</p>
          </AlertDescription>
        </Alert>
      )}

      <fieldset className="flex flex-col gap-4">
        <legend className="mb-1 text-sm font-medium">What you pay</legend>

        <div className="flex flex-col gap-2">
          <label htmlFor="compression-price-preset" className="text-sm font-medium">
            Price list
          </label>
          <Select
            value={priceId}
            onValueChange={(next) => {
              const preset = findPricePreset(next)
              if (!preset) return
              update({
                inputPrice: String(preset.inputPrice),
                cachedInputPrice: String(preset.cachedInputPrice),
                outputPrice: String(preset.outputPrice),
              })
            }}
          >
            <SelectTrigger id="compression-price-preset" className="w-full">
              <SelectValue>{pricePreset?.label ?? 'Your own rates'}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {PRICE_PRESETS.map((preset) => (
                <SelectItem key={preset.id} value={preset.id}>
                  <span className="flex w-full items-center justify-between gap-4">
                    <Marquee>{preset.label}</Marquee>
                    <span className="text-muted-foreground shrink-0 text-xs tabular-nums">
                      ${preset.inputPrice} / ${preset.cachedInputPrice} / ${preset.outputPrice}
                    </span>
                  </span>
                </SelectItem>
              ))}
              <SelectItem value={CUSTOM_PRICE_ID}>Your own rates</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-muted-foreground text-xs">
            {pricePreset?.note ?? 'List rates change, so type the three rates from your own bill.'}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <NumberField
            id="compression-input-price"
            label="Input price"
            hint="Dollars for 1M fresh tokens."
            decimal
            min={0}
            value={inputs.inputPrice}
            onChange={(inputPrice) => update({ inputPrice })}
            invalid={invalidField === 'inputPrice'}
          />
          <NumberField
            id="compression-cached-price"
            label="Cached input price"
            hint="Dollars for 1M tokens read from the cache."
            decimal
            min={0}
            value={inputs.cachedInputPrice}
            onChange={(cachedInputPrice) => update({ cachedInputPrice })}
            invalid={invalidField === 'cachedInputPrice'}
          />
          <NumberField
            id="compression-output-price"
            label="Output price"
            hint="Dollars for 1M tokens written back."
            decimal
            min={0}
            value={inputs.outputPrice}
            onChange={(outputPrice) => update({ outputPrice })}
            invalid={invalidField === 'outputPrice'}
          />
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="mb-1 text-sm font-medium">How your sessions spend tokens</legend>

        <SegmentedControl
          legend="Session type"
          options={MIX_OPTIONS}
          value={mixId}
          wrap
          onValueChange={(next) => {
            const preset = findWorkloadPreset(next)
            if (!preset) return
            update({
              missPercent: String(preset.mix.missPercent),
              cachePercent: String(preset.mix.cachePercent),
              outputPercent: String(preset.mix.outputPercent),
            })
          }}
        />

        <div className="grid gap-4 sm:grid-cols-3">
          <NumberField
            id="compression-miss"
            label="Miss share"
            hint="Percent of the billed tokens that miss the cache."
            decimal
            min={0}
            value={inputs.missPercent}
            onChange={(missPercent) => update({ missPercent })}
            invalid={invalidField === 'missPercent'}
          />
          <NumberField
            id="compression-cache"
            label="Cache share"
            hint="Percent of the billed tokens that hit the cache."
            decimal
            min={0}
            value={inputs.cachePercent}
            onChange={(cachePercent) => update({ cachePercent })}
            invalid={invalidField === 'cachePercent'}
          />
          <NumberField
            id="compression-output-share"
            label="Output share"
            hint="Percent of the billed tokens the reply writes."
            decimal
            min={0}
            value={inputs.outputPercent}
            onChange={(outputPercent) => update({ outputPercent })}
            invalid={invalidField === 'outputPercent'}
          />
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="mb-1 text-sm font-medium">How far you compress</legend>

        <NumberField
          id="compression-percent"
          label="Context kept"
          hint="Percent of the session the summary keeps."
          decimal
          min={1}
          value={inputs.compressionPercent}
          onChange={(compressionPercent) => update({ compressionPercent })}
          invalid={invalidField === 'compressionPercent'}
        />

        <input
          id="compression-percent-range"
          type="range"
          min={1}
          max={100}
          step={1}
          value={sliderValue}
          aria-label="Context kept, in percent"
          onChange={(event) => update({ compressionPercent: event.target.value })}
          className="accent-primary w-full"
        />
      </fieldset>
    </form>
  )
}
