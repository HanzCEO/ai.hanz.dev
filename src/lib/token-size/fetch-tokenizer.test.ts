import { describe, expect, it } from 'vitest'

import { fetchTokenizer } from './fetch-tokenizer'
import { TokenizerError } from './types'

/** A tiny byte-level tokenizer.json the analysis can actually read. */
const TOKENIZER_BODY = JSON.stringify({
  pre_tokenizer: { type: 'ByteLevel' },
  model: {
    vocab: { H: 0, e: 1, l: 2, o: 3, 'Ġ': 4, w: 5, r: 6, d: 7, Hello: 8, 'Ġworld': 9 },
    merges: ['H e', 'He l', 'Hel l', 'Hell o', 'Ġ w', 'Ġw o', 'Ġwo r', 'Ġwor l', 'Ġworl d'],
  },
})

interface Captured {
  url: string
  headers: Record<string, string>
}

function mockFetch(options: { status?: number; body?: string }) {
  const captured: Captured[] = []
  const impl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const headers = (init?.headers ?? {}) as Record<string, string>
    captured.push({ url, headers })
    return new Response(options.body ?? '', {
      status: options.status ?? 200,
      headers: { 'content-type': 'application/json' },
    })
  }) as typeof fetch
  return { impl, captured }
}

describe('fetchTokenizer', () => {
  it('reads tokenizer.json from the HuggingFace resolve path', async () => {
    const { impl, captured } = mockFetch({ body: TOKENIZER_BODY })
    const { profile, url } = await fetchTokenizer({
      provider: 'huggingface',
      repo: 'Qwen/Qwen3-8B',
      corpus: 'Hello world',
      fetchImpl: impl,
    })

    expect(url).toBe('https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json')
    expect(profile.family).toBe('byte_level_bpe')
    expect(profile.corpusTokens).toBe(2)
    expect(profile.bytesPerToken).toBeCloseTo(11 / 2, 9)
    expect(captured[0].headers.Authorization).toBeUndefined()
  })

  it('reads tokenizer.json from the ModelScope resolve path', async () => {
    const { impl, captured } = mockFetch({ body: TOKENIZER_BODY })
    await fetchTokenizer({
      provider: 'modelscope',
      repo: 'Qwen/Qwen3-8B',
      fetchImpl: impl,
    })

    expect(captured[0].url).toBe(
      'https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/tokenizer.json',
    )
  })

  it('sends a bearer token only when one is given', async () => {
    const { impl, captured } = mockFetch({ body: TOKENIZER_BODY })
    await fetchTokenizer({
      provider: 'huggingface',
      repo: 'meta-llama/Llama-3.1-8B-Instruct',
      token: 'hf_abc',
      fetchImpl: impl,
    })

    expect(captured[0].headers.Authorization).toBe('Bearer hf_abc')
  })

  it('turns a missing tokenizer into a not found error', async () => {
    const { impl } = mockFetch({ status: 404 })
    await expect(
      fetchTokenizer({ provider: 'huggingface', repo: 'org/nope', fetchImpl: impl }),
    ).rejects.toThrow(/no tokenizer\.json for org\/nope/)
  })

  it('names gated repos and what to do about them', async () => {
    const { impl } = mockFetch({ status: 401, body: 'gated repo' })
    await expect(
      fetchTokenizer({ provider: 'huggingface', repo: 'org/gated', fetchImpl: impl }),
    ).rejects.toThrow(/gated/)
  })

  it('rejects an id that is not owner/name', async () => {
    await expect(fetchTokenizer({ provider: 'huggingface', repo: 'not-an-id' })).rejects.toThrow(
      /should be owner\/name/,
    )
  })

  it('wraps an unreadable file as an unsupported error', async () => {
    const { impl } = mockFetch({ body: '{}' })
    const error = await fetchTokenizer({
      provider: 'huggingface',
      repo: 'org/empty',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(TokenizerError)
    expect((error as TokenizerError).kind).toBe('unsupported')
    expect((error as TokenizerError).message).toContain('No vocabulary found')
  })

  it('treats non-JSON as malformed', async () => {
    const { impl } = mockFetch({ body: '<html>not json</html>' })
    const error = await fetchTokenizer({
      provider: 'huggingface',
      repo: 'org/html',
      fetchImpl: impl,
    }).catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(TokenizerError)
    expect((error as TokenizerError).kind).toBe('malformed')
  })
})