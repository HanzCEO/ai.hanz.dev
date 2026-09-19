import { Eye, EyeOff, Loader2 } from 'lucide-react'
import { useState } from 'react'

import { PROVIDER_LIST, type ComputeResult, type Provider } from '@/lib/kvcache'
import { Input } from '@/components/ui/input'
import DtypeSelect from '@/tools/kv-cache-calculator/DtypeSelect'
import { MODEL_PRESETS } from '@/tools/kv-cache-calculator/presets'
import type { ModelConfigState } from '@/tools/kv-cache-calculator/useModelConfig'
import type { CalculatorInputs } from '@/tools/kv-cache-calculator/useCalculatorState'

const PROVIDER_ORDER: Provider[] = ['huggingface', 'modelscope']

interface CalculatorFormProps {
  inputs: CalculatorInputs
  update: (patch: Partial<CalculatorInputs>) => void
  configState: ModelConfigState
  result: ComputeResult | null
  contextError: string | null
  sequenceError: string | null
  onUseMaxContext: (max: number) => void
}

export default function CalculatorForm({
  inputs,
  update,
  configState,
  result,
  contextError,
  sequenceError,
  onUseMaxContext,
}: CalculatorFormProps) {
  const [showToken, setShowToken] = useState(false)

  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)
  const detected = configState.status === 'ready' ? result : null
  const showIndexer = detected?.indexerDtypeSupport != null

  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
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
        <label htmlFor="model-id" className="text-sm font-medium">
          Model id
        </label>
        <Input
          id="model-id"
          list="model-presets"
          value={inputs.modelId}
          onChange={(event) => update({ modelId: event.target.value })}
          placeholder="owner/name"
          spellCheck={false}
          autoComplete="off"
        />
        <datalist id="model-presets">
          {MODEL_PRESETS.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.note}
            </option>
          ))}
        </datalist>
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          {configState.status === 'loading' && (
            <>
              <Loader2 className="size-3 animate-spin" aria-hidden="true" />
              Reading the config
            </>
          )}
          {configState.status === 'idle' && 'Type a model id, or pick a suggestion.'}
          {configState.status === 'ready' && detected && (
            <>
              {detected.architecture.label} · {detected.layerSplit.total} layers ·{' '}
              {detected.architecture.modelType}
            </>
          )}
          {configState.status === 'error' && 'Could not read that config.'}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <label htmlFor="context-length" className="text-sm font-medium">
            Context length
          </label>
          <Input
            id="context-length"
            inputMode="numeric"
            type="number"
            value={inputs.contextLength}
            onChange={(event) => update({ contextLength: event.target.value })}
            aria-invalid={contextError !== null}
            aria-describedby={contextError ? 'context-length-error' : undefined}
          />
          {contextError ? (
            <p id="context-length-error" className="text-xs text-rose-600 dark:text-rose-400">
              {contextError}
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              {detected?.maxPositionEmbeddings ? (
                <button
                  type="button"
                  className="underline underline-offset-4 hover:text-foreground"
                  onClick={() => onUseMaxContext(detected.maxPositionEmbeddings as number)}
                >
                  Use the model max, {detected.maxPositionEmbeddings.toLocaleString('en-US')}
                </button>
              ) : (
                'Tokens per sequence.'
              )}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="sequence-count" className="text-sm font-medium">
            Sequences
          </label>
          <Input
            id="sequence-count"
            inputMode="numeric"
            type="number"
            value={inputs.sequenceCount}
            onChange={(event) => update({ sequenceCount: event.target.value })}
            aria-invalid={sequenceError !== null}
            aria-describedby={sequenceError ? 'sequence-count-error' : undefined}
          />
          {sequenceError ? (
            <p id="sequence-count-error" className="text-xs text-rose-600 dark:text-rose-400">
              {sequenceError}
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">Concurrent requests in the batch.</p>
          )}
        </div>
      </div>

      <DtypeSelect
        id="kv-cache-dtype"
        label="KV cache dtype"
        value={inputs.kvCacheDtype}
        onValueChange={(value) => update({ kvCacheDtype: value })}
        support={detected?.dtypeSupport ?? []}
      />

      {showIndexer && detected?.indexerDtypeSupport && (
        <DtypeSelect
          id="indexer-dtype"
          label="Indexer dtype"
          value={inputs.indexerDtype}
          onValueChange={(value) => update({ indexerDtype: value })}
          support={detected.indexerDtypeSupport}
        />
      )}

      <div className="flex flex-col gap-2">
        <label htmlFor="provider-token" className="text-sm font-medium">
          {providerSpec?.tokenLabel ?? 'token'}
        </label>
        <div className="relative">
          <Input
            id="provider-token"
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
    </form>
  )
}
