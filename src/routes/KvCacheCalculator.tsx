import { useCallback, useEffect, useMemo, useRef } from 'react'
import { Link } from 'react-router'

import { computeKvCache, detectArchitecture, unwrapConfig, type ComputeResult, type Provider } from '@/lib/kvcache'
import CalculatorForm from '@/tools/kv-cache-calculator/CalculatorForm'
import CalculatorResults from '@/tools/kv-cache-calculator/CalculatorResults'
import { defaultDtypesForFamily } from '@/tools/kv-cache-calculator/defaultDtypes'
import {
  parsePositiveInteger,
  useCalculatorState,
} from '@/tools/kv-cache-calculator/useCalculatorState'
import { useModelConfig } from '@/lib/use-model-config'

export default function KvCacheCalculator() {
  const { inputs, update, seeded } = useCalculatorState()
  const configState = useModelConfig(inputs.provider, inputs.modelId, inputs.token)

  const contextLength = parsePositiveInteger(inputs.contextLength)
  const sequenceCount = parsePositiveInteger(inputs.sequenceCount)

  const contextError =
    contextLength === null ? 'Enter a whole number of tokens, 1 or more.' : null
  const sequenceError = sequenceCount === null ? 'Enter a whole number, 1 or more.' : null

  const config = configState.status === 'ready' ? configState.config : null

  const detectedFamily = useMemo(() => {
    if (!config) return null
    const { inner, outer } = unwrapConfig(config)
    return detectArchitecture(inner, outer).family
  }, [config])

  const { result, computeError } = useMemo((): {
    result: ComputeResult | null
    computeError: string | null
  } => {
    if (!config || contextLength === null || sequenceCount === null) {
      return { result: null, computeError: null }
    }
    try {
      return {
        result: computeKvCache(config, {
          contextLength,
          sequenceCount,
          kvCacheDtype: inputs.kvCacheDtype,
          indexerDtype: inputs.indexerDtype,
        }),
        computeError: null,
      }
    } catch (error) {
      return {
        result: null,
        computeError:
          error instanceof Error
            ? error.message
            : 'This model config could not be turned into a size.',
      }
    }
  }, [config, contextLength, sequenceCount, inputs.kvCacheDtype, inputs.indexerDtype])

  // Give the context field a sensible value for the model that was detected,
  // but never overwrite a value the user or a shared link supplied.
  const autoSized = useRef(false)
  useEffect(() => {
    if (autoSized.current) return
    if (configState.status !== 'ready') return
    autoSized.current = true

    if (seeded.current.context) return
    const max = result?.maxPositionEmbeddings
    if (max && max > 0) {
      update({ contextLength: String(max) })
    }
  }, [configState.status, result, seeded, update])

  // Open a newly selected model on the cache dtype it actually ships with. The
  // ref keys on the selection, so a manual dtype change is never undone while
  // the same model stays selected, and a shared link that names a dtype keeps
  // its own choice.
  const dtypeSeededFor = useRef<string | null>(null)
  useEffect(() => {
    if (!detectedFamily) return
    const selection = `${inputs.provider}:${inputs.modelId}`
    if (dtypeSeededFor.current === selection) return
    dtypeSeededFor.current = selection
    if (seeded.current.dtype) return

    const next = defaultDtypesForFamily(detectedFamily)
    update({ kvCacheDtype: next.kvCacheDtype, indexerDtype: next.indexerDtype })
  }, [detectedFamily, inputs.provider, inputs.modelId, seeded, update])

  const onSwitchProvider = useCallback(
    (provider: Provider) => update({ provider }),
    [update],
  )

  const onUseMaxContext = useCallback(
    (max: number) => update({ contextLength: String(max) }),
    [update],
  )

  return (
    <div className="flex flex-col gap-8">
      <nav className="text-muted-foreground text-xs">
        <Link to="/" className="underline-offset-4 hover:underline">
          Tools
        </Link>
        <span aria-hidden="true"> / </span>
        <span>KV Cache Calculator</span>
      </nav>

      <header className="flex max-w-2xl flex-col gap-3">
        <h1 className="text-2xl font-medium tracking-tight">KV Cache Calculator</h1>
        <p className="text-muted-foreground">
          Size the cache for a model, a context length, and a sequence count. The config is read
          from HuggingFace or ModelScope.
        </p>
      </header>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)] lg:gap-12">
        <CalculatorForm
          inputs={inputs}
          update={update}
          configState={configState}
          result={result}
          contextError={contextError}
          sequenceError={sequenceError}
          onUseMaxContext={onUseMaxContext}
        />

        <CalculatorResults
          result={result}
          configState={configState}
          onSwitchProvider={onSwitchProvider}
          computeError={computeError}
        />
      </div>
    </div>
  )
}
