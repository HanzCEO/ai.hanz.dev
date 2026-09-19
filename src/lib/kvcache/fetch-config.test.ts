import { describe, expect, it, vi } from 'vitest'

import { PROVIDERS, fetchModelConfig, isValidRepoId, normalizeRepoId } from './fetch-config'
import { ModelConfigError } from './types'

interface Captured {
  url: string
  headers: Record<string, string>
}

function mockFetch(options: { status?: number; body?: string; throws?: Error }) {
  const captured: Captured[] = []

  const impl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const headers = (init?.headers ?? {}) as Record<string, string>
    captured.push({ url, headers })

    if (options.throws) throw options.throws

    return new Response(options.body ?? '', {
      status: options.status ?? 200,
      headers: { 'content-type': 'application/json' },
    })
  }) as typeof fetch

  return { impl, captured }
}

describe('normalizeRepoId', () => {
  it('accepts a bare repo id', () => {
    expect(normalizeRepoId('Qwen/Qwen3-8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeRepoId('  Qwen/Qwen3-8B  ')).toBe('Qwen/Qwen3-8B')
  })

  it('strips a HuggingFace URL', () => {
    expect(normalizeRepoId('https://huggingface.co/Qwen/Qwen3-8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeRepoId('https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json')).toBe(
      'Qwen/Qwen3-8B',
    )
    expect(normalizeRepoId('https://huggingface.co/Qwen/Qwen3-8B/tree/main')).toBe('Qwen/Qwen3-8B')
    expect(normalizeRepoId('https://huggingface.co/Qwen/Qwen3-8B/')).toBe('Qwen/Qwen3-8B')
  })

  it('strips a ModelScope URL', () => {
    expect(normalizeRepoId('https://modelscope.cn/models/Qwen/Qwen3-8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeRepoId('https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master')).toBe(
      'Qwen/Qwen3-8B',
    )
  })

  it('returns an empty string for blank input', () => {
    expect(normalizeRepoId('   ')).toBe('')
  })
})

describe('isValidRepoId', () => {
  it('accepts owner/name and rejects anything else', () => {
    expect(isValidRepoId('Qwen/Qwen3-8B')).toBe(true)
    expect(isValidRepoId('deepseek-ai/DeepSeek-V3.2-Exp')).toBe(true)
    expect(isValidRepoId('Qwen3-8B')).toBe(false)
    expect(isValidRepoId('a/b/c')).toBe(false)
    expect(isValidRepoId('')).toBe(false)
    expect(isValidRepoId('/Qwen3-8B')).toBe(false)
  })
})

describe('endpoint selection', () => {
  it('builds the documented URLs', () => {
    expect(PROVIDERS.huggingface.configUrl('Qwen/Qwen3-8B')).toBe(
      'https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json',
    )
    // The HF shaped ModelScope path is the only one that sends CORS headers.
    expect(PROVIDERS.modelscope.configUrl('Qwen/Qwen3-8B')).toBe(
      'https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/config.json',
    )
  })

  it('labels the token field per provider', () => {
    expect(PROVIDERS.huggingface.tokenLabel).toBe('hf_token')
    expect(PROVIDERS.modelscope.tokenLabel).toBe('ms_token')
  })
})

describe('fetchModelConfig', () => {
  it('returns the parsed config and the URL it used', async () => {
    const { impl, captured } = mockFetch({ body: '{"model_type":"qwen3"}' })
    const result = await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: impl,
    })

    expect(result.config).toEqual({ model_type: 'qwen3' })
    expect(result.url).toBe('https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json')
    expect(captured[0].url).toBe(result.url)
  })

  it('normalizes a pasted URL before fetching', async () => {
    const { impl, captured } = mockFetch({ body: '{}' })
    await fetchModelConfig({
      provider: 'huggingface',
      repo: 'https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json',
      fetchImpl: impl,
    })
    expect(captured[0].url).toBe('https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json')
  })

  it('sends an Authorization header only when a token is supplied', async () => {
    const without = mockFetch({ body: '{}' })
    await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: without.impl,
    })
    expect(without.captured[0].headers.Authorization).toBeUndefined()

    const withToken = mockFetch({ body: '{}' })
    await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      token: 'hf_abc123',
      fetchImpl: withToken.impl,
    })
    expect(withToken.captured[0].headers.Authorization).toBe('Bearer hf_abc123')
  })

  it('ignores a blank token', async () => {
    const { impl, captured } = mockFetch({ body: '{}' })
    await fetchModelConfig({
      provider: 'modelscope',
      repo: 'Qwen/Qwen3-8B',
      token: '   ',
      fetchImpl: impl,
    })
    expect(captured[0].headers.Authorization).toBeUndefined()
  })

  it('trims a token with surrounding whitespace', async () => {
    const { impl, captured } = mockFetch({ body: '{}' })
    await fetchModelConfig({
      provider: 'modelscope',
      repo: 'Qwen/Qwen3-8B',
      token: '  ms_xyz  ',
      fetchImpl: impl,
    })
    expect(captured[0].headers.Authorization).toBe('Bearer ms_xyz')
  })
})

describe('error handling', () => {
  it('explains a gated HuggingFace repo and asks for a token', async () => {
    const { impl } = mockFetch({
      status: 401,
      body: 'Access to model meta-llama/Llama-3.1-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.',
    })

    const error = await fetchModelConfig({
      provider: 'huggingface',
      repo: 'meta-llama/Llama-3.1-8B-Instruct',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(ModelConfigError)
    const configError = error as ModelConfigError
    expect(configError.kind).toBe('gated')
    expect(configError.message).toMatch(/gated/i)
    expect(configError.message).toMatch(/token/i)
  })

  it('reports a missing HuggingFace repo', async () => {
    const { impl } = mockFetch({ status: 404, body: 'Not Found' })
    const error = (await fetchModelConfig({
      provider: 'huggingface',
      repo: 'nope/does-not-exist',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('not_found')
    expect(error.message).toMatch(/no config\.json/i)
    expect(error.suggestOtherProvider).toBe(false)
  })

  it('parses the ModelScope JSON error envelope', async () => {
    const { impl } = mockFetch({
      status: 404,
      body: JSON.stringify({
        Code: 10990101007,
        Message: '获取模型文件失败，文件内容为空',
        Success: false,
      }),
    })

    const error = (await fetchModelConfig({
      provider: 'modelscope',
      repo: 'zai-org/GLM-4.5',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('not_found')
    expect(error.message).toContain('获取模型文件失败')
    // ModelScope mirrors are incomplete, so point at the other provider.
    expect(error.suggestOtherProvider).toBe(true)
  })

  it('treats a 200 with a ModelScope failure envelope as an error', async () => {
    const { impl } = mockFetch({
      status: 200,
      body: JSON.stringify({ Code: 10010101, Message: 'model not exist', Success: false }),
    })

    const error = (await fetchModelConfig({
      provider: 'modelscope',
      repo: 'Qwen/NotReal',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error).toBeInstanceOf(ModelConfigError)
    expect(error.message).toContain('model not exist')
  })

  it('reports a network failure without leaking a raw exception', async () => {
    const { impl } = mockFetch({ throws: new TypeError('Failed to fetch') })
    const error = (await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error).toBeInstanceOf(ModelConfigError)
    expect(error.kind).toBe('network')
    expect(error.message).toMatch(/could not reach/i)
  })

  it('rejects a non JSON body', async () => {
    const { impl } = mockFetch({ status: 200, body: '<html>nope</html>' })
    const error = (await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('malformed')
    expect(error.message).toMatch(/not JSON/i)
  })

  it('rejects a JSON body that is not an object', async () => {
    const { impl } = mockFetch({ status: 200, body: '[1,2,3]' })
    const error = (await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('malformed')
  })

  it('rejects a blank repo id before making a request', async () => {
    const { impl, captured } = mockFetch({ body: '{}' })
    const error = (await fetchModelConfig({
      provider: 'huggingface',
      repo: '   ',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('malformed')
    expect(captured).toHaveLength(0)
  })

  it('rejects a repo id that is not owner/name', async () => {
    const { impl, captured } = mockFetch({ body: '{}' })
    const error = (await fetchModelConfig({
      provider: 'modelscope',
      repo: 'just-a-name',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('malformed')
    expect(error.message).toMatch(/owner\/name/i)
    expect(captured).toHaveLength(0)
  })

  it('surfaces an unexpected server error with the status', async () => {
    const { impl } = mockFetch({ status: 500, body: 'boom' })
    const error = (await fetchModelConfig({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)) as ModelConfigError

    expect(error.kind).toBe('unknown')
    expect(error.message).toContain('500')
  })

  it('rethrows an abort rather than reporting it as a network failure', async () => {
    const abortError = new DOMException('aborted', 'AbortError')
    const { impl } = mockFetch({ throws: abortError })

    await expect(
      fetchModelConfig({
        provider: 'huggingface',
        repo: 'Qwen/Qwen3-8B',
        fetchImpl: impl,
      }),
    ).rejects.toBe(abortError)
  })

  it('never throws a bare Error for any failure mode', async () => {
    const cases = [
      mockFetch({ status: 401, body: 'restricted' }),
      mockFetch({ status: 403, body: 'forbidden' }),
      mockFetch({ status: 404, body: 'missing' }),
      mockFetch({ status: 429, body: 'slow down' }),
      mockFetch({ status: 500, body: 'boom' }),
      mockFetch({ status: 200, body: 'not json' }),
      mockFetch({ throws: new TypeError('Failed to fetch') }),
    ]

    for (const provider of ['huggingface', 'modelscope'] as const) {
      for (const entry of cases) {
        const error = await fetchModelConfig({
          provider,
          repo: 'Qwen/Qwen3-8B',
          fetchImpl: entry.impl,
        }).catch((caught: unknown) => caught)

        expect(error).toBeInstanceOf(ModelConfigError)
        expect((error as ModelConfigError).message.length).toBeGreaterThan(10)
      }
    }
  })
})

describe('no live network access', () => {
  it('never calls the global fetch when an implementation is injected', async () => {
    const spy = vi.spyOn(globalThis, 'fetch')
    const { impl } = mockFetch({ body: '{}' })

    await fetchModelConfig({ provider: 'huggingface', repo: 'Qwen/Qwen3-8B', fetchImpl: impl })

    expect(spy).not.toHaveBeenCalled()
    spy.mockRestore()
  })
})
