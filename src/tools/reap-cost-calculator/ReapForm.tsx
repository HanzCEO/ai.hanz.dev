import { Eye, EyeOff, Loader2 } from 'lucide-react'
import { useState } from 'react'

import { STORAGE_PRESETS, type GpuSpec } from '@/lib/hardware'
import { PROVIDER_LIST, type Provider } from '@/lib/kvcache'
import {
  CALIBRATION_PRESETS,
  PRUNE_RATIOS,
  findCalibrationPreset,
  type WeightDtype,
} from '@/lib/reap'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

import GpuSelect from './GpuSelect'
import WeightDtypeSelect from './WeightDtypeSelect'
import type { ReapShapeState } from './useReapShape'
import type { InputMode, ReapFormInputs } from './useReapState'

const PROVIDER_ORDER: Provider[] = ['huggingface', 'modelscope']

const MODE_LABELS: Array<{ id: InputMode; label: string; hint: string }> = [
  { id: 'hub', label: 'Model id', hint: 'Read the config from HuggingFace or ModelScope.' },
  { id: 'paste', label: 'Paste config', hint: 'Paste a config.json you already have.' },
  { id: 'manual', label: 'Enter numbers', hint: 'Describe the model by hand.' },
]

/** The preset whose sample count and length match the current inputs. */
function activePresetId(samples: string, sequenceLength: string): string {
  const match = CALIBRATION_PRESETS.find(
    (preset) => String(preset.samples) === samples && String(preset.sequenceLength) === sequenceLength,
  )
  return match?.id ?? 'custom'
}

function Field({
  id,
  label,
  hint,
  value,
  onChange,
  invalid,
}: {
  id: string
  label: string
  hint?: string
  value: string
  onChange: (value: string) => void
  invalid?: boolean
}) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <Input
        id={id}
        inputMode="numeric"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-invalid={invalid}
      />
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

interface ReapFormProps {
  inputs: ReapFormInputs
  update: (patch: Partial<ReapFormInputs>) => void
  shapeState: ReapShapeState
  gpu: GpuSpec
}

export default function ReapForm({ inputs, update, shapeState, gpu }: ReapFormProps) {
  const [showToken, setShowToken] = useState(false)
  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)
  const presetId = activePresetId(inputs.samples, inputs.sequenceLength)
  const preset = findCalibrationPreset(presetId)

  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <fieldset className="flex flex-col gap-2">
        <legend className="mb-2 text-sm font-medium">Model source</legend>
        <div className="border-border inline-flex w-fit flex-wrap rounded-lg border p-0.5">
          {MODE_LABELS.map((mode) => {
            const active = inputs.mode === mode.id
            return (
              <button
                key={mode.id}
                type="button"
                aria-pressed={active}
                onClick={() => update({ mode: mode.id })}
                className={`rounded-md px-3 py-1 text-sm transition-colors ${
                  active
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {mode.label}
              </button>
            )
          })}
        </div>
        <p className="text-xs text-muted-foreground">
          {MODE_LABELS.find((mode) => mode.id === inputs.mode)?.hint}
        </p>
      </fieldset>

      {inputs.mode === 'hub' && (
        <>
          <fieldset className="flex flex-col gap-2">
            <legend className="mb-2 text-sm font-medium">Provider</legend>
            <div className="border-border inline-flex w-fit rounded-lg border p-0.5">
              {PROVIDER_ORDER.map((provider) => {
                const spec = PROVIDER_LIST.find((entry) => entry.id === provider)
                const active = inputs.provider === provider
                return (
                  <button
                    key={provider}
                    type="button"
                    aria-pressed={active}
                    onClick={() => update({ provider })}
                    className={`rounded-md px-3 py-1 text-sm transition-colors ${
                      active
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {spec?.label ?? provider}
                  </button>
                )
              })}
            </div>
          </fieldset>

          <div className="flex flex-col gap-2">
            <label htmlFor="reap-model-id" className="text-sm font-medium">
              Model id
            </label>
            <Input
              id="reap-model-id"
              value={inputs.modelId}
              onChange={(event) => update({ modelId: event.target.value })}
              placeholder="owner/name"
              spellCheck={false}
              autoComplete="off"
            />
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              {shapeState.status === 'loading' && (
                <>
                  <Loader2 className="size-3 animate-spin" aria-hidden="true" />
                  Reading the config
                </>
              )}
              {shapeState.status === 'idle' && 'Type a model id.'}
              {shapeState.status === 'ready' && shapeState.shape && (
                <>
                  {shapeState.shape.modelType} · {shapeState.shape.numLayers} layers ·{' '}
                  {shapeState.shape.moeLayers} with experts
                </>
              )}
              {shapeState.status === 'not-moe' && 'That model has no expert bank.'}
              {shapeState.status === 'error' && 'Could not read that config.'}
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="reap-token" className="text-sm font-medium">
              {providerSpec?.tokenLabel ?? 'token'}
            </label>
            <div className="relative">
              <Input
                id="reap-token"
                type={showToken ? 'text' : 'password'}
                value={inputs.token}
                onChange={(event) => update({ token: event.target.value })}
                placeholder="Optional"
                spellCheck={false}
                autoComplete="off"
                className="pr-9"
              />
              <button
                type="button"
                onClick={() => setShowToken((current) => !current)}
                aria-label={showToken ? 'Hide the token' : 'Show the token'}
                className="text-muted-foreground hover:text-foreground absolute inset-y-0 right-0 flex w-9 items-center justify-center"
              >
                {showToken ? (
                  <EyeOff className="size-4" aria-hidden="true" />
                ) : (
                  <Eye className="size-4" aria-hidden="true" />
                )}
              </button>
            </div>
            <p className="text-xs text-muted-foreground">{providerSpec?.tokenHint}</p>
          </div>
        </>
      )}

      {inputs.mode === 'paste' && (
        <div className="flex flex-col gap-2">
          <label htmlFor="reap-config-text" className="text-sm font-medium">
            config.json
          </label>
          <textarea
            id="reap-config-text"
            value={inputs.configText}
            onChange={(event) => update({ configText: event.target.value })}
            spellCheck={false}
            placeholder='{ "model_type": "qwen3_moe", "num_hidden_layers": 48, ... }'
            className="border-input focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 min-h-40 w-full rounded-lg border bg-transparent px-2.5 py-2 font-mono text-xs outline-none focus-visible:ring-3"
          />
          <p className="text-xs text-muted-foreground">
            Nothing is uploaded. The config is read in the browser.
          </p>
        </div>
      )}

      {inputs.mode === 'manual' && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            id="reap-hidden"
            label="hidden_size"
            value={inputs.hiddenSize}
            onChange={(hiddenSize) => update({ hiddenSize })}
          />
          <Field
            id="reap-moe-ffn"
            label="moe_intermediate_size"
            hint="Width of one expert."
            value={inputs.moeIntermediateSize}
            onChange={(moeIntermediateSize) => update({ moeIntermediateSize })}
          />
          <Field
            id="reap-experts"
            label="routed experts"
            value={inputs.routedExperts}
            onChange={(routedExperts) => update({ routedExperts })}
          />
          <Field
            id="reap-topk"
            label="experts per token"
            hint="The router top-k."
            value={inputs.expertsPerToken}
            onChange={(expertsPerToken) => update({ expertsPerToken })}
          />
          <Field
            id="reap-layers"
            label="num_hidden_layers"
            value={inputs.numLayers}
            onChange={(numLayers) => update({ numLayers })}
          />
          <Field
            id="reap-moe-layers"
            label="layers with experts"
            hint="The rest are dense MLPs."
            value={inputs.moeLayers}
            onChange={(moeLayers) => update({ moeLayers })}
          />
          <Field
            id="reap-shared"
            label="shared experts"
            hint="Always-on experts. Zero for most models."
            value={inputs.sharedExperts}
            onChange={(sharedExperts) => update({ sharedExperts })}
          />
          <Field
            id="reap-ffn"
            label="intermediate_size"
            hint="Dense MLP width, used by non-expert layers."
            value={inputs.intermediateSize}
            onChange={(intermediateSize) => update({ intermediateSize })}
          />
          <Field
            id="reap-vocab"
            label="vocab_size"
            value={inputs.vocabSize}
            onChange={(vocabSize) => update({ vocabSize })}
          />
          <Field
            id="reap-heads"
            label="attention heads"
            value={inputs.attentionHeads}
            onChange={(attentionHeads) => update({ attentionHeads })}
          />
          <Field
            id="reap-head-dim"
            label="head_dim"
            value={inputs.headDim}
            onChange={(headDim) => update({ headDim })}
          />
          <Field
            id="reap-kv-heads"
            label="key and value heads"
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
                    <span>
                      {entry.label} · {entry.samples.toLocaleString('en-US')} x{' '}
                      {entry.sequenceLength.toLocaleString('en-US')}
                    </span>
                    <span className="text-muted-foreground text-xs">{entry.note}</span>
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
          <Field
            id="reap-samples"
            label="Samples"
            value={inputs.samples}
            onChange={(samples) => update({ samples })}
          />
          <Field
            id="reap-seq"
            label="Tokens per sample"
            value={inputs.sequenceLength}
            onChange={(sequenceLength) => update({ sequenceLength })}
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
            {gpu.vramGiB} GB of memory, {gpu.bandwidthGBs.toLocaleString('en-US')} GB/s. {gpu.note}
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
            Where the weights are read from
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
                    <span>{entry.label}</span>
                    <span className="text-muted-foreground text-xs tabular-nums">
                      {entry.bandwidthGBs} GB/s
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            The layer-wise observer streams one block at a time, so a slow disk can become the limit.
          </p>
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Assumptions</legend>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            id="reap-mfu"
            label="GPU utilisation"
            hint="A share of dense peak, between 0 and 1. A third is realistic."
            value={inputs.mfu}
            onChange={(mfu) => update({ mfu })}
          />
          <Field
            id="reap-overhead"
            label="Overhead factor"
            hint="Framework, observer, and dataloader cost over the raw figure."
            value={inputs.overheadFactor}
            onChange={(overheadFactor) => update({ overheadFactor })}
          />
          <Field
            id="reap-micro-batch"
            label="Micro batch"
            hint="Samples in flight at once. Drives the activation buffer."
            value={inputs.microBatchSize}
            onChange={(microBatchSize) => update({ microBatchSize })}
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
              Off by default. Pruning alone shrinks the model in memory but leaves the arithmetic per
              token unchanged.
            </p>
          </div>
        </div>
      </fieldset>
    </form>
  )
}
