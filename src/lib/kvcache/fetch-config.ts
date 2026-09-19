import { ModelConfigError, type Provider, type RawConfig } from './types'

export interface ProviderSpec {
  id: Provider
  label: string
  /** Form field name shown to the user. */
  tokenLabel: string
  tokenHint: string
  configUrl: (repo: string) => string
}

/**
 * Both endpoints are reachable from the browser. HuggingFace echoes the Origin
 * back in access-control-allow-origin, and ModelScope's resolve path returns a
 * wildcard. ModelScope's other file endpoint, /api/v1/models/.../repo, sends no
 * CORS headers at all, so it cannot be used from a static page.
 */
export const PROVIDERS: Record<Provider, ProviderSpec> = {
  huggingface: {
    id: 'huggingface',
    label: 'HuggingFace',
    tokenLabel: 'hf_token',
    tokenHint: 'Only needed for gated or private repos. Sent to HuggingFace and nowhere else.',
    configUrl: (repo) => `https://huggingface.co/${repo}/resolve/main/config.json`,
  },
  modelscope: {
    id: 'modelscope',
    label: 'ModelScope',
    tokenLabel: 'ms_token',
    tokenHint: 'Only needed for gated or private repos. Sent to ModelScope and nowhere else.',
    configUrl: (repo) => `https://modelscope.cn/models/${repo}/resolve/master/config.json`,
  },
}

export const PROVIDER_LIST: ProviderSpec[] = [PROVIDERS.huggingface, PROVIDERS.modelscope]

/** Accepts a bare repo id or a pasted URL and returns the bare repo id. */
export function normalizeRepoId(input: string): string {
  let value = input.trim()

  value = value.replace(/^https?:\/\/(www\.)?(huggingface\.co|modelscope\.cn)\//i, '')
  value = value.replace(/^(models|datasets)\//i, '')
  value = value.replace(/^\//, '')

  // Drop any trailing path such as /resolve/main/config.json or /tree/main.
  value = value.replace(/\/(resolve|tree|blob|raw)\/.*$/i, '')
  value = value.replace(/\/(config\.json)$/i, '')

  // Collapse any remaining trailing slashes.
  value = value.replace(/\/+$/, '')

  return value
}

export function isValidRepoId(repo: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(repo)
}

interface ModelScopeError {
  Code?: number | string
  Message?: string
  Success?: boolean
}

function parseModelScopeError(text: string): ModelScopeError | null {
  try {
    const parsed = JSON.parse(text) as unknown
    if (parsed && typeof parsed === 'object' && 'Message' in parsed) {
      return parsed as ModelScopeError
    }
  } catch {
    /* not JSON */
  }
  return null
}

function buildError(options: {
  provider: Provider
  repo: string
  status: number
  body: string
}): ModelConfigError {
  const { provider, repo, status, body } = options
  const trimmed = body.trim()

  if (provider === 'huggingface') {
    if (status === 401 || status === 403) {
      const detail = trimmed.slice(0, 200)
      const gated = /restricted|gated|authenticat|access to model/i.test(trimmed)
      return new ModelConfigError(
        gated
          ? `${repo} is gated. Add a HuggingFace token that has access to it.`
          : `${repo} returned ${status}. Add a HuggingFace token with access to this repo.`,
        { kind: 'gated', provider, repo, cause: detail },
      )
    }
    if (status === 404) {
      return new ModelConfigError(
        `No config.json found for ${repo} on HuggingFace. Check the repo id.`,
        { kind: 'not_found', provider, repo },
      )
    }
    return new ModelConfigError(
      `HuggingFace returned ${status} for ${repo}.${trimmed ? ` ${trimmed.slice(0, 160)}` : ''}`,
      { kind: 'unknown', provider, repo },
    )
  }

  // ModelScope reports failures in a JSON envelope, sometimes with a 200.
  const envelope = parseModelScopeError(trimmed)
  const message = envelope?.Message

  if (status === 404) {
    return new ModelConfigError(
      message
        ? `ModelScope has no config.json for ${repo}. It said: ${message}`
        : `ModelScope has no config.json for ${repo}.`,
      { kind: 'not_found', provider, repo, suggestOtherProvider: true },
    )
  }

  if (status === 401 || status === 403) {
    return new ModelConfigError(
      message ?? `${repo} on ModelScope needs a token with access.`,
      { kind: 'gated', provider, repo },
    )
  }

  return new ModelConfigError(
    message
      ? `ModelScope rejected the request for ${repo}: ${message}`
      : `ModelScope returned ${status} for ${repo}.`,
    { kind: 'unknown', provider, repo },
  )
}

export interface FetchModelConfigOptions {
  provider: Provider
  repo: string
  token?: string
  /** Injected for tests. */
  fetchImpl?: typeof fetch
  signal?: AbortSignal
}

/**
 * Fetches a model's config.json and returns it as a plain object. Every failure
 * path resolves to a ModelConfigError with a message worth showing a user.
 */
export async function fetchModelConfig(
  options: FetchModelConfigOptions,
): Promise<{ config: RawConfig; url: string }> {
  const { provider, token, signal } = options
  const fetchImpl = options.fetchImpl ?? fetch
  const spec = PROVIDERS[provider]
  const repo = normalizeRepoId(options.repo)

  if (!repo) {
    throw new ModelConfigError('Enter a model id, for example Qwen/Qwen3-8B.', {
      kind: 'malformed',
      provider,
      repo,
    })
  }

  if (!isValidRepoId(repo)) {
    throw new ModelConfigError(
      `${repo} does not look like a model id. It should be owner/name, for example Qwen/Qwen3-8B.`,
      { kind: 'malformed', provider, repo },
    )
  }

  const url = spec.configUrl(repo)
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (token && token.trim()) {
    headers.Authorization = `Bearer ${token.trim()}`
  }

  let response: Response
  try {
    response = await fetchImpl(url, { headers, signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ModelConfigError(
      `Could not reach ${spec.label}. This is usually a network problem or a blocked request.`,
      { kind: 'network', provider, repo, cause: error },
    )
  }

  let text: string
  try {
    text = await response.text()
  } catch (error) {
    throw new ModelConfigError(`Could not read the response from ${spec.label}.`, {
      kind: 'network',
      provider,
      repo,
      cause: error,
    })
  }

  if (!response.ok) {
    throw buildError({ provider, repo, status: response.status, body: text })
  }

  // ModelScope can answer 200 with a failure envelope.
  const envelope = provider === 'modelscope' ? parseModelScopeError(text) : null
  if (envelope && envelope.Success === false) {
    throw buildError({ provider, repo, status: 404, body: text })
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new ModelConfigError(
      `${spec.label} returned something that is not JSON. The repo may not hold a model config.`,
      { kind: 'malformed', provider, repo, cause: error },
    )
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new ModelConfigError(`${spec.label} returned an unexpected config shape.`, {
      kind: 'malformed',
      provider,
      repo,
    })
  }

  return { config: parsed as RawConfig, url }
}
