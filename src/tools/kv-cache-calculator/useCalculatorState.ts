import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'

import { DTYPES, type DtypeId, type Provider } from '@/lib/kvcache'

export interface CalculatorInputs {
  provider: Provider
  modelId: string
  /** Kept as text so the field can be emptied while typing. */
  contextLength: string
  sequenceCount: string
  kvCacheDtype: DtypeId
  indexerDtype: DtypeId
  /** Never written to the URL. */
  token: string
}

export const DEFAULT_MODEL_ID = 'Qwen/Qwen3-8B'
export const DEFAULT_CONTEXT_LENGTH = '32768'
export const DEFAULT_SEQUENCE_COUNT = '1'

const DEFAULTS: CalculatorInputs = {
  provider: 'huggingface',
  modelId: DEFAULT_MODEL_ID,
  contextLength: DEFAULT_CONTEXT_LENGTH,
  sequenceCount: DEFAULT_SEQUENCE_COUNT,
  kvCacheDtype: 'BF16',
  indexerDtype: 'BF16',
  token: '',
}

const DTYPE_IDS = new Set<string>(DTYPES.map((dtype) => dtype.id))

function isDtype(value: string | null): value is DtypeId {
  return value !== null && DTYPE_IDS.has(value)
}

function isProvider(value: string | null): value is Provider {
  return value === 'huggingface' || value === 'modelscope'
}

function digitsOnly(value: string | null): string | null {
  if (value === null) return null
  const trimmed = value.trim()
  return /^\d+$/.test(trimmed) ? trimmed : null
}

/**
 * Inputs live in the query string so a calculation can be shared or reloaded.
 * The URL is only read after mount, which keeps the first client render
 * identical to the prerendered HTML and avoids a hydration mismatch.
 */
export function useCalculatorState() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [inputs, setInputs] = useState<CalculatorInputs>(DEFAULTS)
  const mounted = useRef(false)
  // Records which fields the URL supplied, so the page does not overwrite a
  // shared link with its own idea of a sensible default.
  const seeded = useRef({ context: false, model: false, dtype: false })

  useEffect(() => {
    if (mounted.current) return
    mounted.current = true

    const provider = searchParams.get('provider')
    const model = searchParams.get('model')
    const context = digitsOnly(searchParams.get('context'))
    const sequences = digitsOnly(searchParams.get('sequences'))
    const kvDtype = searchParams.get('kv_dtype')
    const indexerDtype = searchParams.get('indexer_dtype')

    seeded.current = {
      context: context !== null,
      model: model !== null && model.trim() !== '',
      dtype: isDtype(kvDtype) || isDtype(indexerDtype),
    }

    setInputs((current) => ({
      ...current,
      provider: isProvider(provider) ? provider : current.provider,
      modelId: model && model.trim() ? model.trim() : current.modelId,
      contextLength: context ?? current.contextLength,
      sequenceCount: sequences ?? current.sequenceCount,
      kvCacheDtype: isDtype(kvDtype) ? kvDtype : current.kvCacheDtype,
      indexerDtype: isDtype(indexerDtype) ? indexerDtype : current.indexerDtype,
    }))
    // Only the first mount should seed from the URL.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const serialized = useMemo(() => {
    const params = new URLSearchParams()
    if (inputs.provider !== DEFAULTS.provider) params.set('provider', inputs.provider)
    if (inputs.modelId !== DEFAULTS.modelId) params.set('model', inputs.modelId)
    if (inputs.contextLength !== DEFAULTS.contextLength) params.set('context', inputs.contextLength)
    if (inputs.sequenceCount !== DEFAULTS.sequenceCount) {
      params.set('sequences', inputs.sequenceCount)
    }
    if (inputs.kvCacheDtype !== DEFAULTS.kvCacheDtype) params.set('kv_dtype', inputs.kvCacheDtype)
    if (inputs.indexerDtype !== DEFAULTS.indexerDtype) {
      params.set('indexer_dtype', inputs.indexerDtype)
    }
    return params.toString()
  }, [inputs])

  useEffect(() => {
    if (!mounted.current) return
    if (serialized === searchParams.toString()) return
    setSearchParams(new URLSearchParams(serialized), { replace: true, preventScrollReset: true })
  }, [serialized, searchParams, setSearchParams])

  const update = useCallback((patch: Partial<CalculatorInputs>) => {
    setInputs((current) => ({ ...current, ...patch }))
  }, [])

  return { inputs, update, serialized, seeded }
}

/** Parses a text field into a positive integer, or null when it is not one. */
export function parsePositiveInteger(value: string): number | null {
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  if (!Number.isSafeInteger(parsed) || parsed < 1) return null
  return parsed
}
