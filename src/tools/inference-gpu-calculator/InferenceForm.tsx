import type { InferenceInputField, InferencePrecision } from '@/lib/inference'
import NumberField from '@/components/ui/number-field'
import SegmentedControl from '@/components/ui/segmented'

import type { InferenceFormInputs } from './useInferenceState'

/**
 * The two precisions, with the one fact that decides the answer.
 *
 * Both take two bytes for each weight, so the picker cannot change the memory
 * figure. It is offered because a reader arrives with a precision in mind, and
 * the hint says plainly that the two give the same result.
 */
const PRECISION_OPTIONS: Array<{ id: InferencePrecision; label: string; hint: string }> = [
  {
    id: 'FP16',
    label: 'FP16',
    hint: 'Two bytes for each weight. A narrow exponent range, so a large activation can overflow.',
  },
  {
    id: 'BF16',
    label: 'BF16',
    hint: 'Two bytes for each weight. A wide exponent range, so it is the safer choice for a large model. The memory footprint is the same as FP16.',
  },
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
 * cannot know: the precision the weights are served in, and the memory limits.
 */
export default function InferenceForm({
  inputs,
  update,
  invalidField,
  headroomError,
  maxGpusError,
}: InferenceFormProps) {
  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <SegmentedControl
        legend="Precision"
        options={PRECISION_OPTIONS}
        value={inputs.precision}
        onValueChange={(precision) => update({ precision })}
        wrap
      />

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
