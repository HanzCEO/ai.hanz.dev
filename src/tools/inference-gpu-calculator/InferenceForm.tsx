import {
  INFERENCE_WEIGHT_FORMATS,
  MTP_HEADS,
  type InferenceInputField,
} from '@/lib/inference'
import { getWeightFormat, weightFormatLabel, type WeightFormatId } from '@/lib/weight-format'
import NumberField from '@/components/ui/number-field'
import SegmentedControl from '@/components/ui/segmented'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

import type { InferenceFormInputs, InferenceWeightChoice } from './useInferenceState'

/** The bytes for each weight, in the short form the hint uses. */
function bytesHint(id: WeightFormatId): string {
  const spec = getWeightFormat(id)
  if (spec.blockSize === null) {
    return spec.bytes === 1 ? '1 byte for each weight' : `${spec.bytes} bytes for each weight`
  }
  return `${spec.bytes.toFixed(4)} bytes for each weight, with the scale sidecar`
}

interface WeightChoice {
  id: InferenceWeightChoice
  label: string
  hint: string
}

/**
 * The automatic choice first, then every format the calculator can price.
 *
 * Auto follows the checkpoint. A checkpoint that stores its experts in MXFP4
 * and the rest in FP8 is therefore costed that way without the reader doing
 * anything. A named format forces one format on the whole model.
 */
const WEIGHT_OPTIONS: WeightChoice[] = [
  {
    id: 'auto',
    label: 'Auto (from the checkpoint)',
    hint: 'The calculator reads the format from the checkpoint config. A mixed checkpoint is costed with its experts apart from the rest.',
  },
  ...INFERENCE_WEIGHT_FORMATS.map((id) => ({
    id,
    label: weightFormatLabel(id),
    hint: bytesHint(id),
  })),
]

interface InferenceFormProps {
  inputs: InferenceFormInputs
  update: (patch: Partial<InferenceFormInputs>) => void
  /** The input the estimator rejected, so it can be marked in the form. */
  invalidField: InferenceInputField | null
  /** Validation message for the headroom, or null. */
  headroomError: string | null
  /** Validation message for the maximum GPU count, or null. */
  maxGpusError: string | null
}

/**
 * The hardware fields.
 *
 * The model, the context length, the sequence count and the cache dtypes belong
 * to step 1, so this form does not repeat them. It asks only for what step 1
 * cannot know: the weight format the checkpoint is served in, the speculative
 * decoding head, and the memory limits.
 */
export default function InferenceForm({
  inputs,
  update,
  invalidField,
  headroomError,
  maxGpusError,
}: InferenceFormProps) {
  const selectedWeight =
    WEIGHT_OPTIONS.find((option) => option.id === inputs.weightFormat) ?? WEIGHT_OPTIONS[0]

  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <div className="flex flex-col gap-2">
        <label htmlFor="inference-weight-format" className="text-sm font-medium">
          Weight format
        </label>
        <Select
          value={inputs.weightFormat}
          onValueChange={(next) => {
            if (WEIGHT_OPTIONS.some((option) => option.id === next)) {
              update({ weightFormat: next as InferenceWeightChoice })
            }
          }}
        >
          <SelectTrigger id="inference-weight-format" className="w-full">
            <SelectValue>{selectedWeight.label}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {WEIGHT_OPTIONS.map((option) => (
              <SelectItem key={option.id} value={option.id}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {selectedWeight.hint}
          {invalidField === 'weightFormat' && ' That format is not one the calculator knows.'}
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <SegmentedControl
          legend="MTP head"
          options={MTP_HEADS.map((head) => ({
            id: head.id,
            label: head.label,
            hint: head.hint,
          }))}
          value={inputs.mtpHead}
          onValueChange={(mtpHead) => update({ mtpHead })}
          wrap
        />
        <p className="text-xs text-muted-foreground">
          MTP is a property of the trained checkpoint. Each model needs its own head. The figure
          here is an estimate for measurement, and not a guarantee.
        </p>
      </div>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Hardware limits</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="inference-headroom"
            label="VRAM headroom"
            hint="A percentage held back for fragmentation and the display. A serving process cannot use the whole card."
            min={0}
            decimal
            value={inputs.headroomPercent}
            onChange={(headroomPercent) => update({ headroomPercent })}
            invalid={invalidField === 'headroom' || headroomError !== null}
          />
          <NumberField
            id="inference-max-gpus"
            label="Maximum GPUs"
            hint="The most cards the answer may use. Raise this to see a configuration that needs more cards."
            min={1}
            value={inputs.maxGpus}
            onChange={(maxGpus) => update({ maxGpus })}
            invalid={invalidField === 'maxGpus' || maxGpusError !== null}
          />
        </div>

        {(headroomError || maxGpusError) && (
          <p className="text-xs text-rose-600 dark:text-rose-400">
            {headroomError ?? maxGpusError}
          </p>
        )}

        <p className="text-xs text-muted-foreground">
          The calculator tests every card in its list at every card count up to this limit. It
          reports the smallest configuration that holds the run.
        </p>
      </fieldset>
    </form>
  )
}
