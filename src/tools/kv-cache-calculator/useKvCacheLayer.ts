import { useCallback, useEffect, useMemo, useRef } from 'react'

import {
  computeKvCache,
  detectArchitecture,
  unwrapConfig,
  type ComputeResult,
  type Provider,
} from '@/lib/kvcache'
import { parsePositiveInteger } from '@/lib/url-state'
import type { ModelConfigState } from '@/lib/use-model-config'
import { defaultDtypesForFamily } from '@/tools/kv-cache-calculator/defaultDtypes'
import type { CalculatorInputs } from '@/tools/kv-cache-calculator/useCalculatorState'

/**
 * The cache half of a calculator, as one hook.
 *
 * The KV cache page and the inference page ask the same first question, so the
 * cache size, its validation, the two defaults that follow from the detected
 * model, and the two callbacks all live here. A page supplies its own inputs,
 * its own update function, and the resolved config.
 */

/** Which cache fields the URL supplied, so a shared link keeps its own values. */
export interface CacheSeededFlags {
  context: boolean
  model: boolean
  dtype: boolean
}

interface UseKvCacheLayerOptions {
  inputs: CalculatorInputs
  update: (patch: Partial<CalculatorInputs>) => void
  configState: ModelConfigState
  /** Read inside the effects, because the URL is applied after the first render. */
  seeded: { readonly current: CacheSeededFlags }
}

export function useKvCacheLayer({
  inputs,
  update,
  configState,
  seeded,
}: UseKvCacheLayerOptions) {
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

  return {
    result,
    computeError,
    contextError,
    sequenceError,
    detectedFamily,
    onSwitchProvider,
    onUseMaxContext,
  }
}
