import { useEffect, useRef, useState } from 'react'

import {
  ModelConfigError,
  fetchModelConfig,
  normalizeRepoId,
  type Provider,
  type RawConfig,
} from '@/lib/kvcache'

/**
 * Fetches and caches a model config from a hub. Shared by every tool that reads
 * a config from HuggingFace or ModelScope.
 */

export type ConfigStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface ModelConfigState {
  status: ConfigStatus
  config: RawConfig | null
  url: string | null
  error: ModelConfigError | null
}

/** Successful fetches are cached so re-selecting a model is instant. */
const cache = new Map<string, { config: RawConfig; url: string }>()

function cacheKey(provider: Provider, repo: string): string {
  return `${provider}:${repo}`
}

const DEBOUNCE_MS = 450

const IDLE: ModelConfigState = { status: 'idle', config: null, url: null, error: null }

function toConfigError(error: unknown, provider: Provider, repo: string): ModelConfigError {
  if (error instanceof ModelConfigError) return error
  return new ModelConfigError('Something went wrong reading the model config.', {
    kind: 'unknown',
    provider,
    repo,
    cause: error,
  })
}

/**
 * Fetches a model config once the provider and model id settle.
 *
 * A cached config resolves during render rather than through an effect, so
 * re-selecting a model costs no extra pass. Failures are never cached, so
 * adding a token retries the same repo.
 */
export function useModelConfig(
  provider: Provider,
  modelId: string,
  token: string,
): ModelConfigState {
  const repo = normalizeRepoId(modelId)
  const key = repo ? cacheKey(provider, repo) : ''

  const [asyncState, setAsyncState] = useState<{ key: string; state: ModelConfigState } | null>(
    null,
  )
  const requestId = useRef(0)

  const cached = repo ? cache.get(key) : undefined

  const resolved: ModelConfigState = !repo
    ? IDLE
    : cached
      ? { status: 'ready', config: cached.config, url: cached.url, error: null }
      : asyncState && asyncState.key === key
        ? asyncState.state
        : { status: 'loading', config: null, url: null, error: null }

  useEffect(() => {
    if (!repo) return
    if (cache.has(key)) return

    const controller = new AbortController()
    const id = ++requestId.current

    const timer = setTimeout(() => {
      setAsyncState({
        key,
        state: { status: 'loading', config: null, url: null, error: null },
      })

      fetchModelConfig({ provider, repo, token, signal: controller.signal })
        .then((result) => {
          if (id !== requestId.current) return
          cache.set(key, result)
          setAsyncState({
            key,
            state: { status: 'ready', config: result.config, url: result.url, error: null },
          })
        })
        .catch((error: unknown) => {
          if (id !== requestId.current) return
          if (error instanceof DOMException && error.name === 'AbortError') return
          setAsyncState({
            key,
            state: {
              status: 'error',
              config: null,
              url: null,
              error: toConfigError(error, provider, repo),
            },
          })
        })
    }, DEBOUNCE_MS)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [provider, repo, key, token])

  return resolved
}
