import { ModelIdField, ProviderPicker, TokenField } from '@/components/model-source/ModelSourceFields'
import { Input } from '@/components/ui/input'
import { PROVIDER_LIST, type ComputeResult } from '@/lib/kvcache'
import DtypeSelect from '@/tools/kv-cache-calculator/DtypeSelect'
import { MODEL_PRESETS } from '@/tools/kv-cache-calculator/presets'
import type { ModelConfigState } from '@/lib/use-model-config'
import type { CalculatorInputs } from '@/tools/kv-cache-calculator/useCalculatorState'

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
  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)
  const detected = configState.status === 'ready' ? result : null
  const showIndexer = detected?.indexerDtypeSupport != null

  return (
    <form className="flex flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <ProviderPicker
        value={inputs.provider}
        onValueChange={(provider) => update({ provider })}
      />

      <ModelIdField
        id="model-id"
        listId="model-presets"
        value={inputs.modelId}
        onChange={(modelId) => update({ modelId })}
        status={configState.status}
        presets={MODEL_PRESETS}
        summary={
          detected && (
            <>
              {detected.architecture.label} · {detected.layerSplit.total} layers ·{' '}
              {detected.architecture.modelType}
            </>
          )
        }
      />

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

      <TokenField
        id="provider-token"
        label={providerSpec?.tokenLabel ?? 'token'}
        value={inputs.token}
        onChange={(token) => update({ token })}
        hint={providerSpec?.tokenHint}
      />
    </form>
  )
}
