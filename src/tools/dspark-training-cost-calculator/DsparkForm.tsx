import type { GpuSpec } from '@/lib/hardware'
import { STORAGE_PRESETS } from '@/lib/hardware'
import {
  DSPARK_PRESETS,
  DEFAULT_GLOBAL_BATCH_SIZE,
  DEFAULT_LEARNING_RATE,
  DEFAULT_WARMUP_RATIO,
  findDsparkPreset,
  presetTrainingTokens,
  type DsparkInputField,
} from '@/lib/dspark'
import { PROVIDER_LIST } from '@/lib/kvcache'
import GpuSelect from '@/components/hardware/GpuSelect'
import {
  ConfigPasteField,
  ModelIdField,
  ProviderPicker,
  TokenField,
} from '@/components/model-source/ModelSourceFields'
import NumberField from '@/components/ui/number-field'
import Marquee from '@/components/ui/marquee'
import SegmentedControl from '@/components/ui/segmented'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

import { activePresetId, type DsparkFormInputs, type InputMode } from './useDsparkState'
import type { DsparkShapeState } from './useDsparkShape'

const MODE_LABELS: Array<{ id: InputMode; label: string; hint: string }> = [
  { id: 'hub', label: 'Model id', hint: 'Read the target config from HuggingFace or ModelScope.' },
  { id: 'paste', label: 'Paste config', hint: 'Paste a target config.json you already have.' },
  { id: 'manual', label: 'Enter numbers', hint: 'Describe the target by hand.' },
]

const DATA_MODE_LABELS = [
  {
    id: 'offline' as const,
    label: 'Precompute the cache',
    hint: 'Runs the target once to write its hidden states to disk, then trains without it. Cheap in VRAM, very expensive in storage.',
  },
  {
    id: 'online' as const,
    label: 'Capture online',
    hint: 'Captures the target hidden states during training and writes nothing. Needs the whole target resident for the entire run.',
  },
]

interface DsparkFormProps {
  inputs: DsparkFormInputs
  update: (patch: Partial<DsparkFormInputs>) => void
  shapeState: DsparkShapeState
  gpu: GpuSpec
  /** The input the estimator rejected, so it can be marked in the form. */
  invalidField: DsparkInputField | null
}

export default function DsparkForm({
  inputs,
  update,
  shapeState,
  gpu,
  invalidField,
}: DsparkFormProps) {
  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)
  const presetId = activePresetId(inputs.samples, inputs.sequenceLength, inputs.epochs)
  const preset = findDsparkPreset(presetId)

  const samples = Number(inputs.samples)
  const sequenceLength = Number(inputs.sequenceLength)
  const tokens =
    Number.isFinite(samples) && Number.isFinite(sequenceLength) ? samples * sequenceLength : 0

  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <SegmentedControl
        legend="Target source"
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
            id="dspark-model-id"
            value={inputs.modelId}
            onChange={(modelId) => update({ modelId })}
            status={shapeState.status}
            idleHint="Type a target model id."
            summary={
              shapeState.shape && (
                <>
                  {shapeState.shape.modelType} · {shapeState.shape.numLayers} layers ·{' '}
                  {shapeState.shape.hiddenSize.toLocaleString('en-US')} hidden
                  {shapeState.shape.moeLayers > 0
                    ? ` · ${shapeState.shape.routedExperts} experts`
                    : ''}
                </>
              )
            }
          />

          <TokenField
            id="dspark-token"
            label={providerSpec?.tokenLabel ?? 'token'}
            value={inputs.token}
            onChange={(token) => update({ token })}
            hint={providerSpec?.tokenHint}
          />
        </>
      )}

      {inputs.mode === 'paste' && (
        <ConfigPasteField
          id="dspark-config-text"
          value={inputs.configText}
          onChange={(configText) => update({ configText })}
          placeholder='{ "model_type": "qwen3", "num_hidden_layers": 36, ... }'
        />
      )}

      {inputs.mode === 'manual' && (
        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="dspark-hidden"
            label="hidden_size"
            min={1}
            value={inputs.hiddenSize}
            onChange={(hiddenSize) => update({ hiddenSize })}
          />
          <NumberField
            id="dspark-ffn"
            label="intermediate_size"
            hint="Dense feed forward width."
            min={1}
            value={inputs.intermediateSize}
            onChange={(intermediateSize) => update({ intermediateSize })}
          />
          <NumberField
            id="dspark-layers"
            label="num_hidden_layers"
            min={1}
            value={inputs.numLayers}
            onChange={(numLayers) => update({ numLayers })}
          />
          <NumberField
            id="dspark-vocab"
            label="vocab_size"
            hint="Sets the Markov head and the shared embedding."
            min={1}
            value={inputs.vocabSize}
            onChange={(vocabSize) => update({ vocabSize })}
          />
          <NumberField
            id="dspark-heads"
            label="num_attention_heads"
            min={1}
            value={inputs.attentionHeads}
            onChange={(attentionHeads) => update({ attentionHeads })}
          />
          <NumberField
            id="dspark-kv-heads"
            label="num_key_value_heads"
            min={1}
            value={inputs.kvHeads}
            onChange={(kvHeads) => update({ kvHeads })}
          />
          <NumberField
            id="dspark-head-dim"
            label="head_dim"
            min={1}
            value={inputs.headDim}
            onChange={(headDim) => update({ headDim })}
          />
          <NumberField
            id="dspark-experts"
            label="routed experts"
            hint="Zero for a dense target."
            min={0}
            value={inputs.routedExperts}
            onChange={(routedExperts) => update({ routedExperts })}
          />
          <NumberField
            id="dspark-topk"
            label="experts per token"
            hint="The router top-k. Ignored on a dense target."
            min={0}
            value={inputs.expertsPerToken}
            onChange={(expertsPerToken) => update({ expertsPerToken })}
          />
          <NumberField
            id="dspark-expert-ffn"
            label="moe_intermediate_size"
            hint="Width of one expert."
            min={0}
            value={inputs.moeIntermediateSize}
            onChange={(moeIntermediateSize) => update({ moeIntermediateSize })}
          />
          <NumberField
            id="dspark-moe-layers"
            label="layers with experts"
            min={0}
            value={inputs.moeLayers}
            onChange={(moeLayers) => update({ moeLayers })}
          />
        </div>
      )}

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Training data</legend>

        <div className="flex flex-col gap-2">
          <label htmlFor="dspark-preset" className="text-sm font-medium">
            Recipe
          </label>
          <Select
            value={presetId}
            onValueChange={(next) => {
              const chosen = findDsparkPreset(next)
              if (chosen) {
                update({
                  samples: String(chosen.samples),
                  sequenceLength: String(chosen.sequenceLength),
                  epochs: String(chosen.epochs),
                  // The anchor count travels with the recipe, because a count
                  // written for long sequences does not fit a short one.
                  numAnchors: String(chosen.numAnchors),
                })
              }
            }}
          >
            <SelectTrigger id="dspark-preset" className="w-full">
              <SelectValue>{preset ? preset.label : 'Custom'}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {DSPARK_PRESETS.map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  <span className="flex w-full flex-col items-start gap-0.5">
                    <Marquee className="w-full">
                      {entry.label} · {entry.samples.toLocaleString('en-US')} x{' '}
                      {entry.sequenceLength.toLocaleString('en-US')} x {entry.epochs}
                    </Marquee>
                    <Marquee className="text-muted-foreground w-full text-xs">{entry.note}</Marquee>
                  </span>
                </SelectItem>
              ))}
              <SelectItem value="custom">Custom</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {preset
              ? `${preset.note} ${presetTrainingTokens(preset).toLocaleString('en-US')} tokens in one pass.`
              : 'Set your own sample count, sequence length and epoch count below.'}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="dspark-samples"
            label="Samples"
            hint="Sequences in the training set."
            min={1}
            value={inputs.samples}
            onChange={(samples) => update({ samples })}
            invalid={invalidField === 'trainingTokens'}
          />
          <NumberField
            id="dspark-seq"
            label="Tokens per sample"
            min={1}
            value={inputs.sequenceLength}
            onChange={(sequenceLength) => update({ sequenceLength })}
            invalid={invalidField === 'sequenceLength'}
          />
          <NumberField
            id="dspark-epochs"
            label="Epochs"
            hint="The paper trains each drafter for 10."
            min={1}
            value={inputs.epochs}
            onChange={(epochs) => update({ epochs })}
            invalid={invalidField === 'epochs'}
          />
        </div>

        <p className="text-xs text-muted-foreground">
          {tokens > 0
            ? `${tokens.toLocaleString('en-US')} tokens in one pass, read ${inputs.epochs || '0'} time${inputs.epochs === '1' ? '' : 's'}. This product, not the sample count alone, is what drives both the cache and the run.`
            : 'Enter a sample count and a sequence length to size the run.'}
        </p>

        <SegmentedControl
          legend="Target supervision"
          options={DATA_MODE_LABELS}
          value={inputs.dataMode}
          onValueChange={(dataMode) => update({ dataMode })}
        />
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Draft recipe</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="dspark-target-layers"
            label="Captured target layers"
            hint="Each one adds a hidden state per token to the cache. Five is the published setting, and fewer is the main way to shrink the cache."
            min={1}
            value={inputs.numTargetLayers}
            onChange={(numTargetLayers) => update({ numTargetLayers })}
            invalid={invalidField === 'numTargetLayers'}
          />
          <NumberField
            id="dspark-draft-layers"
            label="Draft layers"
            hint="Backbone depth. Five matches the published drafters."
            min={1}
            value={inputs.numDraftLayers}
            onChange={(numDraftLayers) => update({ numDraftLayers })}
            invalid={invalidField === 'numDraftLayers'}
          />
          <NumberField
            id="dspark-block"
            label="Block size"
            hint="Tokens drafted per block, the gamma of the paper."
            min={1}
            value={inputs.blockSize}
            onChange={(blockSize) => update({ blockSize })}
            invalid={invalidField === 'blockSize'}
          />
          <NumberField
            id="dspark-anchors"
            label="Anchors per sequence"
            hint="Blocks sampled from each sequence per step. Capped at one block per sequence token, so a short sequence needs fewer."
            min={1}
            value={inputs.numAnchors}
            onChange={(numAnchors) => update({ numAnchors })}
            invalid={invalidField === 'numAnchors'}
          />
          <NumberField
            id="dspark-markov"
            label="Markov rank"
            hint="Rank of the sequential head. Zero disables it, leaving a purely parallel drafter."
            min={0}
            value={inputs.markovRank}
            onChange={(markovRank) => update({ markovRank })}
            invalid={invalidField === 'markovRank'}
          />
        </div>

        <p className="text-xs text-muted-foreground">
          The published recipe trains at a learning rate of {DEFAULT_LEARNING_RATE} with a{' '}
          {DEFAULT_WARMUP_RATIO * 100} percent warmup and an effective batch of{' '}
          {DEFAULT_GLOBAL_BATCH_SIZE}. Neither affects the cost, so neither is an input here.
        </p>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Hardware</legend>

        <div className="flex flex-col gap-2">
          <label htmlFor="dspark-gpu" className="text-sm font-medium">
            GPU
          </label>
          <GpuSelect
            id="dspark-gpu"
            value={inputs.gpuId}
            onValueChange={(gpuId) => update({ gpuId })}
          />
          <p className="text-xs text-muted-foreground">
            {gpu.vramGiB} GB of memory, {gpu.bf16DenseTflops.toLocaleString('en-US')} TFLOPS
            dense in bf16. {gpu.note}
          </p>
        </div>

        <NumberField
          id="dspark-gpu-count"
          label="Cards"
          hint="The run shards across these. The reference configurations assume a single node of eight."
          min={1}
          value={inputs.gpuCount}
          onChange={(gpuCount) => update({ gpuCount })}
          invalid={invalidField === 'gpuCount'}
        />

        <div className="flex flex-col gap-2">
          <label htmlFor="dspark-storage" className="text-sm font-medium">
            Where the target cache lives
          </label>
          <Select
            value={inputs.storageId}
            onValueChange={(storageId) => update({ storageId })}
          >
            <SelectTrigger id="dspark-storage" className="w-full">
              <SelectValue>
                {STORAGE_PRESETS.find((entry) => entry.id === inputs.storageId)?.label ??
                  inputs.storageId}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {STORAGE_PRESETS.map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  <span className="flex w-full items-center justify-between gap-4">
                    <Marquee>{entry.label}</Marquee>
                    <span className="text-muted-foreground shrink-0 text-xs tabular-nums">
                      {entry.bandwidthGBs} GB/s
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {inputs.dataMode === 'offline'
              ? 'The cache is read back once per epoch, so this decides whether the run is bound by the GPU or by the disk.'
              : 'Online capture writes nothing, so this only affects where the target checkpoint is read from.'}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <NumberField
            id="dspark-mfu"
            label="Utilisation"
            hint="Fraction of peak."
            min={0}
            decimal
            value={inputs.mfu}
            onChange={(mfu) => update({ mfu })}
            invalid={invalidField === 'mfu'}
          />
          <NumberField
            id="dspark-overhead"
            label="Overhead"
            hint="Multiplier over the bound."
            min={1}
            decimal
            value={inputs.overheadFactor}
            onChange={(overheadFactor) => update({ overheadFactor })}
            invalid={invalidField === 'overheadFactor'}
          />
          <NumberField
            id="dspark-micro-batch"
            label="Micro batch"
            hint="Sequences in flight."
            min={1}
            value={inputs.microBatchSize}
            onChange={(microBatchSize) => update({ microBatchSize })}
            invalid={invalidField === 'microBatchSize'}
          />
        </div>
      </fieldset>
    </form>
  )
}
