import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'

import { TOKEN_SIZE_FAQ } from '@/lib/token-size'
import { defaultsFromSchema } from '@/lib/url-state'

import TokenSizeCalculator from './TokenSizeCalculator'
import TokenSizeForm from '@/tools/token-size-calculator/TokenSizeForm'
import TokenSizeResults from '@/tools/token-size-calculator/TokenSizeResults'
import {
  TOKEN_SIZE_SCHEMA,
  type TokenSizeInputs,
} from '@/tools/token-size-calculator/useTokenSizeState'

const PATH = '/tools/token-size-calculator/'

/**
 * The page as a server would render it.
 *
 * Effects do not run here, so the tokenizer fetch never fires and this covers
 * the first paint, which is the prerendered HTML a crawler reads.
 */
function renderPage(search = ''): string {
  return renderToString(
    createElement(
      MemoryRouter,
      { initialEntries: [`${PATH}${search}`] },
      createElement(TokenSizeCalculator),
    ),
  )
}

const html = renderPage()

describe('TokenSizeCalculator at the first paint', () => {
  it('opens with the trail back to the tool index and its own heading', () => {
    expect(html).toContain('Token Size Calculator')
    expect(html).toContain('How many GB is a token count?')
    expect(html).toContain('href="/"')
  })

  it('shows both providers and the model id field', () => {
    expect(html).toContain('HuggingFace')
    expect(html).toContain('ModelScope')
    expect(html).toContain('id="token-size-model-id"')
    expect(html).toContain('value="Qwen/Qwen3-8B"')
  })

  it('offers every model preset and token amount preset', () => {
    for (const id of [
      'Qwen/Qwen3-8B',
      'openai-community/gpt2',
      'mistralai/Mistral-7B-v0.1',
      't5-base',
      'bert-base-uncased',
    ]) {
      expect(html).toContain(id)
    }
    for (const label of ['1K', '128K', '1M', '1B', '1T']) {
      expect(html).toContain(label)
    }
  })

  it('reads the default token count of a billion tokens', () => {
    expect(html).toContain('value="1000000000"')
  })

  it('is loading until the tokenizer is read, with no answer printed', () => {
    // On the server the fetch effect never runs, so the page reports that it
    // would read the tokenizer and prints neither an answer nor an error.
    expect(html).toContain('Reading the tokenizer')
    expect(html).toContain('Reading the tokenizer for this model')
    expect(html).not.toContain('data-testid="token-size-headline"')
    expect(html).not.toContain('The same size in every unit')
  })

  it('ignores the query string, because it is applied after mount', () => {
    // This is what keeps the prerendered markup and the client's first render
    // in agreement. The read itself is covered in useTokenSizeState.test.ts.
    const shared = renderPage('?model=org/other&tokens=128k')
    expect(shared).toContain('value="Qwen/Qwen3-8B"')
    expect(shared).toContain('value="1000000000"')
  })

  it('carries the questions the page answers', () => {
    expect(html).toContain('Questions about sizing tokens')
    for (const item of TOKEN_SIZE_FAQ) {
      expect(html).toContain(item.question)
    }
  })
})

describe('TokenSizeResults states', () => {
  function renderResults(patch: Partial<{
    status: 'idle' | 'loading' | 'ready' | 'error'
    profile: unknown
    url: string | null
    error: unknown
    tokensValue: number | null
  }>): string {
    return renderToString(
      createElement(TokenSizeResults, {
        status: 'idle',
        profile: null,
        url: null,
        error: null,
        tokensValue: null,
        ...patch,
        // The props are typed through the concrete component, so patches that
        // are not valid never compile.
      } as Parameters<typeof TokenSizeResults>[0]),
    )
  }

  it('shows the idle hint before the tokenizer resolves', () => {
    expect(renderResults({})).toContain('Pick a model and enter a token count to size it.')
  })

  it('reports the token count error without inventing a size', () => {
    const rejected = renderResults({
      status: 'ready',
      profile: { family: 'byte_level_bpe', vocabSize: 10 },
      tokensValue: null,
    })
    expect(rejected).toContain('The token count is not readable')
  })
})

describe('TokenSizeForm inputs', () => {
  const defaults = defaultsFromSchema(TOKEN_SIZE_SCHEMA)

  function renderForm(patch: Partial<TokenSizeInputs>): string {
    return renderToString(
      createElement(TokenSizeForm, {
        inputs: { ...defaults, ...patch },
        update: () => {},
        status: 'idle',
        profile: null,
        error: null,
      }),
    )
  }

  it('marks the field when the token amount is unusable', () => {
    const invalid = renderForm({ tokens: 'lots' })
    expect(invalid).toContain('aria-invalid="true"')
    expect(invalid).toContain('Enter a whole number of tokens')
  })

  it('leaves the field unmarked for a valid amount', () => {
    const valid = renderForm({ tokens: '128k' })
    expect(valid).not.toContain('aria-invalid="true"')
  })
})