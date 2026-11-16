import { describe, expect, it } from 'vitest'

import { defaultsFromSchema, serializeState } from '@/lib/url-state'

import { DEFAULT_TOKEN_SIZE_MODEL, TOKEN_SIZE_SCHEMA, type TokenSizeInputs } from './useTokenSizeState'

describe('TOKEN_SIZE_SCHEMA defaults', () => {
  it('opens on HuggingFace with Qwen3, a billion tokens, and text storage', () => {
    const defaults = defaultsFromSchema(TOKEN_SIZE_SCHEMA)

    expect(defaults.provider).toBe('huggingface')
    expect(defaults.modelId).toBe(DEFAULT_TOKEN_SIZE_MODEL)
    expect(defaults.token).toBe('')
    expect(defaults.tokens).toBe('1000000000')
    expect(defaults.storage).toBe('text')
  })

  it('serializes nothing when every field is at its default', () => {
    expect(serializeState(TOKEN_SIZE_SCHEMA, defaultsFromSchema(TOKEN_SIZE_SCHEMA))).toBe('')
  })
})

describe('TOKEN_SIZE_SCHEMA URL reading', () => {
  it('accepts a token amount with a suffix from the URL', () => {
    expect(TOKEN_SIZE_SCHEMA.tokens.parse('128k')).toBe('128k')
    expect(TOKEN_SIZE_SCHEMA.tokens.parse('1.5M')).toBe('1.5M')
  })

  it('rejects a token amount the input cannot have produced', () => {
    expect(TOKEN_SIZE_SCHEMA.tokens.parse(null)).toBeNull()
    expect(TOKEN_SIZE_SCHEMA.tokens.parse('not-a-count')).toBeNull()
    expect(TOKEN_SIZE_SCHEMA.tokens.parse('0')).toBeNull()
  })

  it('reads only the providers the page offers', () => {
    expect(TOKEN_SIZE_SCHEMA.provider.parse('modelscope')).toBe('modelscope')
    expect(TOKEN_SIZE_SCHEMA.provider.parse('replicate')).toBeNull()
  })

  it('reads only the storage formats the page offers', () => {
    expect(TOKEN_SIZE_SCHEMA.storage.parse('parquet')).toBe('parquet')
    expect(TOKEN_SIZE_SCHEMA.storage.parse('arrow')).toBe('arrow')
    expect(TOKEN_SIZE_SCHEMA.storage.parse('hdf5')).toBeNull()
  })

  it('writes a non-default storage format beside the token count', () => {
    const defaults = defaultsFromSchema(TOKEN_SIZE_SCHEMA)
    const parquet: TokenSizeInputs = { ...defaults, storage: 'parquet' }
    expect(serializeState(TOKEN_SIZE_SCHEMA, parquet)).toBe('format=parquet')
  })

  it('never writes the hub token to the URL', () => {
    const defaults = defaultsFromSchema(TOKEN_SIZE_SCHEMA)
    expect(
      serializeState(TOKEN_SIZE_SCHEMA, { ...defaults, token: 'hf_secret' }),
    ).toBe('')
  })
})