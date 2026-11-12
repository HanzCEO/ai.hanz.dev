import {
  PROVIDERS,
  isValidRepoId,
  normalizeRepoId,
  type Provider,
} from '@/lib/kvcache'

import { analyzeTokenizer } from './tokenizer'
import { TokenizerError, type TokenizerProfile } from './types'

/**
 * Fetches a tokenizer.json from the same hubs the model config tools read, and
 * analyzes it into a profile the page can size a token count with.
 *
 * The endpoints are the resolve paths the hubs already serve to a browser with
 * CORS. A tokenizer is a bigger file than a config, so the failure messages say
 * what to do next: add a token for a gated repo, check the repo id, or pick a
 * repo that actually publishes a tokenizer.
 */

export interface FetchTokenizerOptions {
  provider: Provider
  repo: string
  token?: string
  /** The text the bytes-per-token ratio is measured over. Defaults to the bundled corpus. */
  corpus?: string
  /** Injected for tests. */
  fetchImpl?: typeof fetch
  signal?: AbortSignal
}

export interface FetchedTokenizer {
  profile: TokenizerProfile
  url: string
}

function tokenizerUrl(provider: Provider, repo: string): string {
  // Same-CORS pair as the config endpoints: HF echoes Origin, ModelScope's
  // resolve path answers with a wildcard.
  if (provider === 'huggingface') {
    return `https://huggingface.co/${repo}/resolve/main/tokenizer.json`
  }
  return `https://modelscope.cn/models/${repo}/resolve/master/tokenizer.json`
}

function buildError(options: {
  provider: Provider
  repo: string
  status: number
  body: string
}): TokenizerError {
  const { provider, repo, status, body } = options
  const label = PROVIDERS[provider].label
  const detail = body.trim().slice(0, 200)

  if (status === 401 || status === 403) {
    const gated = /restricted|gated|authenticat|access to model/i.test(detail)
    return new TokenizerError(
      gated
        ? `${repo} is gated. Add a ${PROVIDERS[provider].tokenLabel} that has access to it.`
        : `${label} refused that tokenizer for ${repo}. Add a token with access, or pick another repo.`,
      { kind: 'gated', provider, repo },
    )
  }

  if (status === 404) {
    return new TokenizerError(
      `${label} has no tokenizer.json for ${repo}. Check the repo id, or pick a model that publishes a tokenizer.`,
      { kind: 'not_found', provider, repo },
    )
  }

  return new TokenizerError(
    detail
      ? `${label} returned ${status} for ${repo}: ${detail}`
      : `${label} returned ${status} for ${repo}.`,
    { kind: 'unknown', provider, repo },
  )
}

/** Fetches and analyzes one model's tokenizer.json. */
export async function fetchTokenizer(options: FetchTokenizerOptions): Promise<FetchedTokenizer> {
  const { provider, token, signal } = options
  const fetchImpl = options.fetchImpl ?? fetch
  const repo = normalizeRepoId(options.repo)
  const label = PROVIDERS[provider].label

  if (!repo) {
    throw new TokenizerError('Enter a model id, for example Qwen/Qwen3-8B.', {
      kind: 'malformed',
      provider,
      repo,
    })
  }

  if (!isValidRepoId(repo)) {
    throw new TokenizerError(
      `${repo} does not look like a model id. It should be owner/name, for example Qwen/Qwen3-8B.`,
      { kind: 'malformed', provider, repo },
    )
  }

  const url = tokenizerUrl(provider, repo)
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (token && token.trim()) headers.Authorization = `Bearer ${token.trim()}`

  let response: Response
  try {
    response = await fetchImpl(url, { headers, signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new TokenizerError(
      `Could not reach ${label}. This is usually a network problem or a blocked request.`,
      { kind: 'network', provider, repo, cause: error },
    )
  }

  let text: string
  try {
    text = await response.text()
  } catch (error) {
    throw new TokenizerError(`Could not read the tokenizer from ${label}.`, {
      kind: 'network',
      provider,
      repo,
      cause: error,
    })
  }

  if (!response.ok) {
    throw buildError({ provider, repo, status: response.status, body: text })
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new TokenizerError(
      `${label} returned something that is not JSON. The repo may not hold a tokenizer.json.`,
      { kind: 'malformed', provider, repo, cause: error },
    )
  }

  try {
    return { profile: analyzeTokenizer(parsed, options.corpus), url }
  } catch (error) {
    throw new TokenizerError(
      error instanceof Error
        ? `${repo} publishes a tokenizer this page cannot read: ${error.message}`
        : `${repo} publishes a tokenizer this page cannot read.`,
      { kind: 'unsupported', provider, repo, cause: error },
    )
  }
}