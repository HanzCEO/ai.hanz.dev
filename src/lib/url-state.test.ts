import { describe, expect, it } from 'vitest'

import {
  defaultsFromSchema,
  digitsOnly,
  nonEmptyText,
  pickEnum,
  serializeState,
  type UrlSchema,
} from './url-state'

describe('pickEnum', () => {
  it('returns the value when it is allowed', () => {
    expect(pickEnum('huggingface', ['huggingface', 'modelscope'])).toBe('huggingface')
  })

  it('returns null for a value outside the list or an absent parameter', () => {
    expect(pickEnum('github', ['huggingface', 'modelscope'])).toBeNull()
    expect(pickEnum(null, ['huggingface'])).toBeNull()
  })
})

describe('digitsOnly', () => {
  it('returns a run of digits, trimmed', () => {
    expect(digitsOnly(' 32768 ')).toBe('32768')
    expect(digitsOnly('0')).toBe('0')
  })

  it('returns null for anything else, including an empty field', () => {
    expect(digitsOnly('')).toBeNull()
    expect(digitsOnly('12a')).toBeNull()
    expect(digitsOnly('-1')).toBeNull()
    expect(digitsOnly(null)).toBeNull()
  })
})

describe('nonEmptyText', () => {
  it('returns trimmed text', () => {
    expect(nonEmptyText('  Qwen/Qwen3-8B ')).toBe('Qwen/Qwen3-8B')
  })

  it('returns null for blank or absent text', () => {
    expect(nonEmptyText('   ')).toBeNull()
    expect(nonEmptyText('')).toBeNull()
    expect(nonEmptyText(null)).toBeNull()
  })
})

interface Sample {
  provider: string
  context: string
  secret: string
}

const SCHEMA: UrlSchema<Sample> = {
  provider: { param: 'provider', default: 'huggingface', parse: pickEnumOf(['huggingface', 'modelscope']) },
  context: { param: 'context', default: '32768', parse: digitsOnly },
  secret: { param: 'secret', default: '', parse: () => null, omit: true },
}

function pickEnumOf(allowed: readonly string[]) {
  return (raw: string | null) => pickEnum(raw, allowed)
}

describe('defaultsFromSchema', () => {
  it('collects each field default', () => {
    expect(defaultsFromSchema(SCHEMA)).toEqual({
      provider: 'huggingface',
      context: '32768',
      secret: '',
    })
  })
})

describe('serializeState', () => {
  it('produces an empty string when every value is at its default', () => {
    expect(serializeState(SCHEMA, defaultsFromSchema(SCHEMA))).toBe('')
  })

  it('writes only the values that differ from their default', () => {
    const query = serializeState(SCHEMA, {
      provider: 'modelscope',
      context: '32768',
      secret: '',
    })
    expect(query).toBe('provider=modelscope')
  })

  it('never writes an omitted field, even when it holds a value', () => {
    const query = serializeState(SCHEMA, {
      provider: 'huggingface',
      context: '32768',
      secret: 'hf_secret_token',
    })
    expect(query).toBe('')
    expect(query).not.toContain('secret')
  })

  it('writes fields in schema order, so the output is stable', () => {
    const query = serializeState(SCHEMA, {
      provider: 'modelscope',
      context: '4096',
      secret: '',
    })
    expect(query).toBe('provider=modelscope&context=4096')
  })

  it('honours a custom serializer', () => {
    const schema: UrlSchema<{ list: string[] }> = {
      list: {
        param: 'list',
        default: [],
        parse: () => null,
        serialize: (value) => value.join(','),
      },
    }
    expect(serializeState(schema, { list: ['a', 'b'] })).toBe('list=a%2Cb')
  })

  it('round-trips through URLSearchParams', () => {
    const values = { provider: 'modelscope', context: '4096', secret: '' }
    const params = new URLSearchParams(serializeState(SCHEMA, values))
    expect(params.get('provider')).toBe('modelscope')
    expect(params.get('context')).toBe('4096')
  })
})
