import { STORAGE_PRESETS, type GpuSpec } from '@/lib/hardware'
import {
  CALIBRATION_PRESETS,
  PRUNE_RATIOS,
  findCalibrationPreset,
  type ReapInputField,
  type WeightDtype,
} from '@/lib/reap'
import NumberField from '@/components/ui/number-field'
import Marquee from '@/components/ui/marquee'
import { PROVIDER_LIST } from '@/lib/kvcache'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import GpuSelect from '@/components/hardware/GpuSelect'
import {
  ConfigPasteField,
  ModelIdField,
  ProviderPicker,
  TokenField,
} from '@/components/model-source/ModelSourceFields'
import SegmentedControl from '@/components/ui/segmented'

import WeightDtypeSelect from './WeightDtypeSelect'
import type { ReapShapeState } from './useReapShape'
import type { InputMode, ReapFormInputs } from './useReapState'

const MODE_LABELS: Array<{ id: InputMode; label: string; hint: string }> = [
  { id: 'hub', label: 'Model id', hint: 'Read the config from HuggingFace or ModelScope.' },
  { id: 'paste', label: 'Paste config', hint: 'Paste a config.json that you already have.' },
  { id: 'manual', label: 'Enter numbers', hint: 'Enter the model values yourself.' },
]

/** The preset whose sample count and length match the current inputs. */
function activePresetId(samples: string, sequenceLength: string): string {
  const match = CALIBRATION_PRESETS.find(
    (preset) => String(preset.samples) === samples && String(preset.sequenceLength) === sequenceLength,
  )
  return match?.id ?? 'custom'
}

interface ReapFormProps {
  inputs: ReapFormInputs
  update: (patch: Partial<ReapFormInputs>) => void
  shapeState: ReapShapeState
  gpu: GpuSpec
  /** The input the estimator rejected, so it can be marked in the form. */
  invalidField: ReapInputField | null
}

export default function ReapForm({ inputs, update, shapeState, gpu, invalidField }: ReapFormProps) {
  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)
  const presetId = activePresetId(inputs.samples, inputs.sequenceLength)
  const preset = findCalibrationPreset(presetId)

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
            id="reap-model-id"
            value={inputs.modelId}
            onChange={(modelId) => update({ modelId })}
            status={shapeState.status}
            idleHint="Type a model id."
            notMoeHint="That model has no expert bank."
            summary={
              shapeState.shape && (
                <>
                  {shapeState.shape.modelType} · {shapeState.shape.numLayers} layers ·{' '}
                  {shapeState.shape.moeLayers} with experts
                </>
              )
            }
          />

          <TokenField
            id="reap-token"
            label={providerSpec?.tokenLabel ?? 'token'}
            value={inputs.token}
            onChange={(token) => update({ token })}
            hint={providerSpec?.tokenHint}
          />
        </>
      )}

      {inputs.mode === 'paste' && (
        <ConfigPasteField
          id="reap-config-text"
          value={inputs.configText}
          onChange={(configText) => update({ configText })}
          placeholder='{ "model_type": "qwen3_moe", "num_hidden_layers": 48, ... }'
        />
      )}

      {inputs.mode === 'manual' && (
        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="reap-hidden"
            label="hidden_size"
            min={1}
            value={inputs.hiddenSize}
            onChange={(hiddenSize) => update({ hiddenSize })}
          />
          <NumberField
            id="reap-moe-ffn"
            label="moe_intermediate_size"
            hint="The width of one expert."
            min={1}
            value={inputs.moeIntermediateSize}
            onChange={(moeIntermediateSize) => update({ moeIntermediateSize })}
          />
          <NumberField
            id="reap-experts"
            label="routed experts"
            min={1}
            value={inputs.routedExperts}
            onChange={(routedExperts) => update({ routedExperts })}
          />
          <NumberField
            id="reap-topk"
            label="experts per token"
            hint="The router top-k."
            min={1}
            value={inputs.expertsPerToken}
            onChange={(expertsPerToken) => update({ expertsPerToken })}
          />
          <NumberField
            id="reap-layers"
            label="num_hidden_layers"
            min={1}
            value={inputs.numLayers}
            onChange={(numLayers) => update({ numLayers })}
          />
          <NumberField
            id="reap-moe-layers"
            label="layers with experts"
            hint="The other blocks have dense MLPs."
            min={1}
            value={inputs.moeLayers}
            onChange={(moeLayers) => update({ moeLayers })}
          />
          <NumberField
            id="reap-shared"
            label="shared experts"
            hint="These experts always run. Most models use 0."
            min={0}
            value={inputs.sharedExperts}
            onChange={(sharedExperts) => update({ sharedExperts })}
          />
          <NumberField
            id="reap-ffn"
            label="intermediate_size"
            hint="The dense MLP width. The blocks without experts use it."
            min={1}
            value={inputs.intermediateSize}
            onChange={(intermediateSize) => update({ intermediateSize })}
          />
          <NumberField
            id="reap-vocab"
            label="vocab_size"
            min={1}
            value={inputs.vocabSize}
            onChange={(vocabSize) => update({ vocabSize })}
          />
          <NumberField
            id="reap-heads"
            label="attention heads"
            min={1}
            value={inputs.attentionHeads}
            onChange={(attentionHeads) => update({ attentionHeads })}
          />
          <NumberField
            id="reap-head-dim"
            label="head_dim"
            min={1}
            value={inputs.headDim}
            onChange={(headDim) => update({ headDim })}
          />
          <NumberField
            id="reap-kv-heads"
            label="key and value heads"
            min={1}
            value={inputs.kvHeads}
            onChange={(kvHeads) => update({ kvHeads })}
          />
        </div>
      )}

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Calibration</legend>

        <div className="flex flex-col gap-2">
          <label htmlFor="reap-preset" className="text-sm font-medium">
            Recipe
          </label>
          <Select
            value={presetId}
            onValueChange={(next) => {
              const chosen = findCalibrationPreset(next)
              if (chosen) {
                update({
                  samples: String(chosen.samples),
                  sequenceLength: String(chosen.sequenceLength),
                })
              }
            }}
          >
            <SelectTrigger id="reap-preset" className="w-full">
              <SelectValue>
                {preset ? preset.label : 'Custom'}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {CALIBRATION_PRESETS.map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  <span className="flex w-full flex-col items-start gap-0.5">
                    <Marquee className="w-full">
                      {entry.label} · {entry.samples.toLocaleString('en-US')} x{' '}
                      {entry.sequenceLength.toLocaleString('en-US')}
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
              ? preset.note
              : 'Set your own sample count and sequence length below.'}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="reap-samples"
            label="Samples"
            min={1}
            value={inputs.samples}
            onChange={(samples) => update({ samples })}
            invalid={invalidField === 'samples'}
          />
          <NumberField
            id="reap-seq"
            label="Tokens per sample"
            min={1}
            value={inputs.sequenceLength}
            onChange={(sequenceLength) => update({ sequenceLength })}
            invalid={invalidField === 'sequenceLength'}
          />
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="reap-ratio" className="text-sm font-medium">
            Pruning ratio
          </label>
          <Select
            value={inputs.pruneRatio}
            onValueChange={(next) => update({ pruneRatio: next })}
          >
            <SelectTrigger id="reap-ratio" className="w-full">
              <SelectValue>{`${Math.round(Number(inputs.pruneRatio) * 100)} percent`}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {PRUNE_RATIOS.map((ratio) => (
                <SelectItem key={ratio} value={String(ratio)}>
                  {`${Math.round(ratio * 100)} percent`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Published checkpoints use 25, 30, 40, and 50 percent.
          </p>
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Hardware</legend>

        <div className="flex flex-col gap-2">
          <label htmlFor="reap-gpu" className="text-sm font-medium">
            GPU
          </label>
          <GpuSelect id="reap-gpu" value={inputs.gpuId} onValueChange={(gpuId) => update({ gpuId })} />
          <p className="text-xs text-muted-foreground">
            This GPU has {gpu.vramGiB} GB of VRAM and {gpu.bandwidthGBs.toLocaleString('en-US')}{' '}
            GB/s of memory bandwidth. {gpu.note}
          </p>
        </div>

        <WeightDtypeSelect
          id="reap-dtype"
          value={inputs.weightDtype}
          gpu={gpu}
          onValueChange={(weightDtype: WeightDtype) => update({ weightDtype })}
        />

        <div className="flex flex-col gap-2">
          <label htmlFor="reap-storage" className="text-sm font-medium">
            Weight storage
          </label>
          <Select
            value={inputs.storageId}
            onValueChange={(next) => update({ storageId: next })}
          >
            <SelectTrigger id="reap-storage" className="w-full">
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
            The layer-wise observer reads one expert block at a time. Slow storage can therefore set
            the duration.
          </p>
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Assumptions</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            id="reap-mfu"
            label="GPU utilisation"
            hint="A share of the dense peak, between 0 and 1. One third is realistic."
            decimal
            value={inputs.mfu}
            onChange={(mfu) => update({ mfu })}
            invalid={invalidField === 'mfu'}
          />
          <NumberField
            id="reap-overhead"
            label="Overhead factor"
            hint="The framework, observer, and dataloader cost over the raw value."
            decimal
            min={1}
            value={inputs.overheadFactor}
            onChange={(overheadFactor) => update({ overheadFactor })}
            invalid={invalidField === 'overheadFactor'}
          />
          <NumberField
            id="reap-micro-batch"
            label="Micro batch"
            hint="The samples in flight at once. This value sets the activation buffer."
            min={1}
            value={inputs.microBatchSize}
            onChange={(microBatchSize) => update({ microBatchSize })}
            invalid={invalidField === 'microBatchSize'}
          />
          <div className="flex flex-col justify-center gap-2">
            <label htmlFor="reap-scale-topk" className="flex items-center gap-2 text-sm font-medium">
              <input
                id="reap-scale-topk"
                type="checkbox"
                checked={inputs.scaleTopK}
                onChange={(event) => update({ scaleTopK: event.target.checked })}
                className="size-4"
              />
              Reduce the router top-k
            </label>
            <p className="text-xs text-muted-foreground">
              This option is off by default. Pruning alone makes the model smaller in VRAM. It does
              not change the arithmetic for each token.
            </p>
          </div>
        </div>
      </fieldset>
    </form>
  )
}
