import { useEffect, useRef, useState } from 'react'

import { normalizeRepoId, type Provider } from '@/lib/kvcache'
import {
  fetchTokenizer,
  TokenizerError,
  type TokenizerProfile,
} from '@/lib/token-size'

/**
 * Fetches and caches one tokenizer profile from a hub.
 *
 * The profile replaces the raw tokenizer.json in the cache, because it is the
 * only thing the page reads: the analysis runs once per repo instead of once
 * per render or per keystroke. A cached profile resolves during render, so
 * re-selecting a model costs no extra pass.
 */

export type TokenizerStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface TokenizerState {
  status: TokenizerStatus
  profile: TokenizerProfile | null
  url: string | null
  error: TokenizerError | null
}

/** Successful fetches are cached by provider and repo. */
const cache = new Map<string, { profile: TokenizerProfile; url: string }>()

function cacheKey(provider: Provider, repo: string): string {
  return `${provider}:${repo}`
}

const DEBOUNCE_MS = 450

const IDLE: TokenizerState = { status: 'idle', profile: null, url: null, error: null }

function toError(error: unknown, provider: Provider, repo: string): TokenizerError {
  if (error instanceof TokenizerError) return error
  return new TokenizerError('Something went wrong reading that tokenizer.', {
    kind: 'unknown',
    provider,
    repo,
    cause: error,
  })
}

export function useTokenizer(
  provider: Provider,
  modelId: string,
  token: string,
): TokenizerState {
  const repo = normalizeRepoId(modelId)
  const key = repo ? cacheKey(provider, repo) : ''

  const [asyncState, setAsyncState] = useState<{ key: string; state: TokenizerState } | null>(
    null,
  )
  const requestId = useRef(0)

  const cached = repo ? cache.get(key) : undefined

  const resolved: TokenizerState = !repo
    ? IDLE
    : cached
      ? { status: 'ready', profile: cached.profile, url: cached.url, error: null }
      : asyncState && asyncState.key === key
        ? asyncState.state
        : { status: 'loading', profile: null, url: null, error: null }

  useEffect(() => {
    if (!repo) return
    if (cache.has(key)) return

    const controller = new AbortController()
    const id = ++requestId.current

    const timer = setTimeout(() => {
      setAsyncState({
        key,
        state: { status: 'loading', profile: null, url: null, error: null },
      })

      fetchTokenizer({ provider, repo, token, signal: controller.signal })
        .then(({ profile, url }) => {
          if (id !== requestId.current) return
          cache.set(key, { profile, url })
          setAsyncState({ key, state: { status: 'ready', profile, url, error: null } })
        })
        .catch((error: unknown) => {
          if (id !== requestId.current) return
          if (error instanceof DOMException && error.name === 'AbortError') return
          setAsyncState({
            key,
            state: { status: 'error', profile: null, url: null, error: toError(error, provider, repo) },
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