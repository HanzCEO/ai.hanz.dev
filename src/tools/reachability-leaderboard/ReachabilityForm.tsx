import { DTYPES } from '@/lib/kvcache'
import { REACHABILITY_GPU_GROUPS, REACHABILITY_SORTS } from '@/lib/reachability'
import NumberField from '@/components/ui/number-field'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

import {
  REACHABILITY_WEIGHT_OPTIONS,
  type ReachabilityErrors,
  type ReachabilityFormInputs,
} from './useReachabilityState'

/** The class choices, with the all cards option in front. */
const CLASS_OPTIONS = [
  { id: 'all' as const, label: 'All cards', hint: 'Every card in the hardware directory.' },
  ...REACHABILITY_GPU_GROUPS.map((group) => ({
    id: group.id,
    label: group.label,
    hint: group.hint,
  })),
]

const VENDOR_OPTIONS = [
  { id: 'all' as const, label: 'All vendors' },
  { id: 'nvidia' as const, label: 'NVIDIA' },
  { id: 'amd' as const, label: 'AMD' },
]

interface ReachabilityFormProps {
  inputs: ReachabilityFormInputs
  update: (patch: Partial<ReachabilityFormInputs>) => void
  errors: ReachabilityErrors
}

/**
 * The workload and the hardware filter.
 *
 * Every control here changes which models can be reached or how the rows are
 * ordered. The table itself does no work, so a reader can see the whole board
 * move as the context length or the card filter changes.
 */
export default function ReachabilityForm({ inputs, update, errors }: ReachabilityFormProps) {
  const selectedClass = CLASS_OPTIONS.find((option) => option.id === inputs.gpuClass)
  const selectedSort = REACHABILITY_SORTS.find((sort) => sort.id === inputs.sort)
  const selectedWeight =
    REACHABILITY_WEIGHT_OPTIONS.find((option) => option.id === inputs.weightFormat) ??
    REACHABILITY_WEIGHT_OPTIONS[0]
  const selectedDtype = DTYPES.find((dtype) => dtype.id === inputs.kvCacheDtype)

  return (
    <form
      className="flex flex-col gap-6"
      aria-label="Reachability workload"
      onSubmit={(event) => event.preventDefault()}
    >
      <div className="flex flex-col gap-2">
        <label htmlFor="reachability-search" className="text-sm font-medium">
          Search
        </label>
        <Input
          id="reachability-search"
          type="search"
          placeholder="Model, engine, publisher, or base checkpoint"
          value={inputs.search}
          onChange={(event) => update({ search: event.target.value })}
        />
        <p className="text-xs text-muted-foreground">
          Matches the model name, the engine name, the publisher, and the base checkpoint.
        </p>
      </div>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Hardware</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label htmlFor="reachability-gpu-class" className="text-sm font-medium">
              Card class
            </label>
            <Select
              value={inputs.gpuClass}
              onValueChange={(next) => {
                const option = CLASS_OPTIONS.find((entry) => entry.id === next)
                if (option) update({ gpuClass: option.id })
              }}
            >
              <SelectTrigger id="reachability-gpu-class" className="w-full">
                <SelectValue>{selectedClass?.label ?? 'All cards'}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {CLASS_OPTIONS.map((option) => (
                  <SelectItem key={option.id} value={option.id}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="reachability-vendor" className="text-sm font-medium">
              Vendor
            </label>
            <Select
              value={inputs.vendor}
              onValueChange={(next) => {
                const option = VENDOR_OPTIONS.find((entry) => entry.id === next)
                if (option) update({ vendor: option.id })
              }}
            >
              <SelectTrigger id="reachability-vendor" className="w-full">
                <SelectValue>
                  {VENDOR_OPTIONS.find((option) => option.id === inputs.vendor)?.label ?? 'All vendors'}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {VENDOR_OPTIONS.map((option) => (
                  <SelectItem key={option.id} value={option.id}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <p className="text-xs text-muted-foreground">
          {selectedClass?.hint ?? CLASS_OPTIONS[0].hint} The card class and the vendor narrow which
          cards the ranking may use, and the table reports the count that still holds each model.
        </p>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Workload</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="reachability-context"
            label="Context length"
            hint="Tokens in each sequence. The KV cache grows with it."
            min={1}
            value={inputs.contextLength}
            onChange={(contextLength) => update({ contextLength })}
            invalid={errors.contextLength !== null}
          />
          <NumberField
            id="reachability-sequences"
            label="Concurrent sequences"
            hint="Sequences served at once. The cache and the activation buffer both grow with it."
            min={1}
            value={inputs.sequences}
            onChange={(sequences) => update({ sequences })}
            invalid={errors.sequences !== null}
          />
        </div>

        {(errors.contextLength || errors.sequences) && (
          <p className="text-xs text-rose-600 dark:text-rose-400">
            {errors.contextLength ?? errors.sequences}
          </p>
        )}

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label htmlFor="reachability-weight-format" className="text-sm font-medium">
              Weight format
            </label>
            <Select
              value={inputs.weightFormat}
              onValueChange={(next) => {
                const option = REACHABILITY_WEIGHT_OPTIONS.find((entry) => entry.id === next)
                if (option) update({ weightFormat: option.id })
              }}
            >
              <SelectTrigger id="reachability-weight-format" className="w-full">
                <SelectValue>{selectedWeight.label}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {REACHABILITY_WEIGHT_OPTIONS.map((option) => (
                  <SelectItem key={option.id} value={option.id}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Published follows each checkpoint. A named format forces one format on every model.
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="reachability-kv-dtype" className="text-sm font-medium">
              KV cache dtype
            </label>
            <Select
              value={inputs.kvCacheDtype}
              onValueChange={(next) => {
                const option = DTYPES.find((dtype) => dtype.id === next)
                if (option) update({ kvCacheDtype: option.id })
              }}
            >
              <SelectTrigger id="reachability-kv-dtype" className="w-full">
                <SelectValue>{selectedDtype?.label ?? inputs.kvCacheDtype}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DTYPES.map((dtype) => (
                  <SelectItem key={dtype.id} value={dtype.id}>
                    {dtype.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              The dtype the cache is held in. A narrower cache lets a smaller card hold a long
              context.
            </p>
          </div>
        </div>

        <NumberField
          id="reachability-max-gpus"
          label="Maximum cards for one model"
          hint="The most cards a configuration may use before the model is reported as not reachable."
          min={1}
          value={inputs.maxGpus}
          onChange={(maxGpus) => update({ maxGpus })}
          invalid={errors.maxGpus !== null}
        />

        {errors.maxGpus && (
          <p className="text-xs text-rose-600 dark:text-rose-400">{errors.maxGpus}</p>
        )}
      </fieldset>

      <div className="flex flex-col gap-2">
        <label htmlFor="reachability-sort" className="text-sm font-medium">
          Order
        </label>
        <Select
          value={inputs.sort}
          onValueChange={(next) => {
            const option = REACHABILITY_SORTS.find((sort) => sort.id === next)
            if (option) update({ sort: option.id })
          }}
        >
          <SelectTrigger id="reachability-sort" className="w-full">
            <SelectValue>{selectedSort?.label ?? 'Reachability'}</SelectValue>
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
          {selectedSort?.hint ?? REACHABILITY_SORTS[0].hint}
        </p>
      </div>
    </form>
  )
}
