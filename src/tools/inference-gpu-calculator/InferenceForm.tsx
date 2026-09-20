import { PROVIDER_LIST } from '@/lib/kvcache'
import type { InferenceInputField, InferencePrecision } from '@/lib/inference'
import {
  ConfigPasteField,
  ModelIdField,
  ProviderPicker,
  TokenField,
} from '@/components/model-source/ModelSourceFields'
import NumberField from '@/components/ui/number-field'
import SegmentedControl from '@/components/ui/segmented'

import { MODEL_PRESETS } from './presets'
import type { InferenceShapeState } from './useInferenceShape'
import type { InferenceFormInputs, InputMode } from './useInferenceState'

const MODE_LABELS: Array<{ id: InputMode; label: string; hint: string }> = [
  { id: 'hub', label: 'Model id', hint: 'Read the config from HuggingFace or ModelScope.' },
  { id: 'paste', label: 'Paste config', hint: 'Paste a config.json that you already have.' },
  { id: 'manual', label: 'Enter numbers', hint: 'Enter the model values yourself.' },
]

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

const TIE_OPTIONS: Array<{ id: InferenceFormInputs['tieEmbeddings']; label: string; hint: string }> =
  [
    {
      id: 'untied',
      label: 'Separate head',
      hint: 'The embedding and the language model head are 2 separate tables. Most models do this.',
    },
    {
      id: 'tied',
      label: 'Tied head',
      hint: 'The embedding and the language model head share 1 table, so the checkpoint is smaller by 1 embedding table.',
    },
  ]

interface InferenceFormProps {
  inputs: InferenceFormInputs
  update: (patch: Partial<InferenceFormInputs>) => void
  shapeState: InferenceShapeState
  /** The input the estimator rejected, so it can be marked in the form. */
  invalidField: InferenceInputField | null
  /** Validation message for the context length, or null. */
  contextError: string | null
  /** Validation message for the sequence count, or null. */
  sequenceError: string | null
  /** Validation message for the headroom, or null. */
  headroomError: string | null
  /** Validation message for the maximum GPU count, or null. */
  maxGpusError: string | null
}

export default function InferenceForm({
  inputs,
  update,
  shapeState,
  invalidField,
  contextError,
  sequenceError,
  headroomError,
  maxGpusError,
}: InferenceFormProps) {
  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)
  const shape = shapeState.shape

  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <SegmentedControl
        legend="Model source"
        options={MODE_LABELS}
        value={inputs.mode}
        onValueChange={(mode) => update({ mode })}
        wrap
      />

      {inputs.mode === 'hub' && (
        <>
          <ProviderPicker
            value={inputs.provider}
            onValueChange={(provider) => update({ provider })}
          />

          <ModelIdField
            id="inference-model-id"
            listId="inference-model-presets"
            value={inputs.modelId}
            onChange={(modelId) => update({ modelId })}
            status={shapeState.status}
            presets={MODEL_PRESETS}
            idleHint="Type a model id for the model you want to serve."
            errorHint="The calculator cannot read that config."
            summary={
              shape && (
                <>
                  {shape.modelType} · {shape.numLayers} layers ·{' '}
                  {shape.hiddenSize.toLocaleString('en-US')} hidden
                  {shape.moeLayers > 0
                    ? ` · ${shape.routedExperts} experts, ${shape.expertsPerToken} for each token`
                    : ''}
                </>
              )
            }
          />

          <TokenField
            id="inference-token"
            label={providerSpec?.tokenLabel ?? 'token'}
            value={inputs.token}
            onChange={(token) => update({ token })}
            hint={providerSpec?.tokenHint}
          />
        </>
      )}

      {inputs.mode === 'paste' && (
        <ConfigPasteField
          id="inference-config-text"
          value={inputs.configText}
          onChange={(configText) => update({ configText })}
          placeholder='{ "model_type": "qwen3", "num_hidden_layers": 36, ... }'
        />
      )}

      {inputs.mode === 'manual' && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            <NumberField
              id="inference-hidden"
              label="hidden_size"
              min={1}
              value={inputs.hiddenSize}
              onChange={(hiddenSize) => update({ hiddenSize })}
            />
            <NumberField
              id="inference-ffn"
              label="intermediate_size"
              hint="The width of the dense feed forward block."
              min={1}
              value={inputs.intermediateSize}
              onChange={(intermediateSize) => update({ intermediateSize })}
            />
            <NumberField
              id="inference-layers"
              label="num_hidden_layers"
              min={1}
              value={inputs.numLayers}
              onChange={(numLayers) => update({ numLayers })}
            />
            <NumberField
              id="inference-vocab"
              label="vocab_size"
              hint="This value sets the size of the embedding table."
              min={1}
              value={inputs.vocabSize}
              onChange={(vocabSize) => update({ vocabSize })}
            />
            <NumberField
              id="inference-heads"
              label="num_attention_heads"
              min={1}
              value={inputs.attentionHeads}
              onChange={(attentionHeads) => update({ attentionHeads })}
            />
            <NumberField
              id="inference-kv-heads"
              label="num_key_value_heads"
              hint="Fewer than the attention heads means grouped query attention, so the cache is smaller."
              min={1}
              value={inputs.kvHeads}
              onChange={(kvHeads) => update({ kvHeads })}
            />
            <NumberField
              id="inference-head-dim"
              label="head_dim"
              min={1}
              value={inputs.headDim}
              onChange={(headDim) => update({ headDim })}
            />
            <NumberField
              id="inference-experts"
              label="routed experts"
              hint="Use 0 for a dense model."
              min={0}
              value={inputs.routedExperts}
              onChange={(routedExperts) => update({ routedExperts })}
            />
            <NumberField
              id="inference-topk"
              label="experts per token"
              hint="The router top-k. This value sets the weights one token reads."
              min={0}
              value={inputs.expertsPerToken}
              onChange={(expertsPerToken) => update({ expertsPerToken })}
            />
            <NumberField
              id="inference-expert-ffn"
              label="moe_intermediate_size"
              hint="The width of one expert."
              min={0}
              value={inputs.moeIntermediateSize}
              onChange={(moeIntermediateSize) => update({ moeIntermediateSize })}
            />
            <NumberField
              id="inference-moe-layers"
              label="layers with experts"
              min={0}
              value={inputs.moeLayers}
              onChange={(moeLayers) => update({ moeLayers })}
            />
          </div>

          <SegmentedControl
            legend="Embedding and head"
            options={TIE_OPTIONS}
            value={inputs.tieEmbeddings}
            onValueChange={(tieEmbeddings) => update({ tieEmbeddings })}
            wrap
          />
        </>
      )}

      <SegmentedControl
        legend="Precision"
        options={PRECISION_OPTIONS}
        value={inputs.precision}
        onValueChange={(precision) => update({ precision })}
        wrap
      />

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">What you serve</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="inference-context"
            label="Context length"
            hint="The tokens in each sequence. The cache grows in proportion."
            min={1}
            value={inputs.contextLength}
            onChange={(contextLength) => update({ contextLength })}
            invalid={invalidField === 'contextLength' || contextError !== null}
          />
          <NumberField
            id="inference-sequences"
            label="Concurrent sequences"
            hint="The requests served at the same time. Every one holds its own cache."
            min={1}
            value={inputs.sequences}
            onChange={(sequences) => update({ sequences })}
            invalid={invalidField === 'sequences' || sequenceError !== null}
          />
        </div>

        {(contextError || sequenceError) && (
          <p className="text-xs text-rose-600 dark:text-rose-400">
            {contextError ?? sequenceError}
          </p>
        )}
      </fieldset>

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
