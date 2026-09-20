import { describe, expect, it } from 'vitest'

import {
  booleanFlag,
  decimalOrNull,
  defaultsFromSchema,
  digitsOrNull,
  enumOf,
  nonEmptyText,
  parsePositiveInteger,
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

describe('enumOf', () => {
  const parse = enumOf(['huggingface', 'modelscope'] as const)

  it('returns the value when it is allowed', () => {
    expect(parse('modelscope')).toBe('modelscope')
  })

  it('returns null for a value outside the list or an absent parameter', () => {
    expect(parse('github')).toBeNull()
    expect(parse(null)).toBeNull()
    expect(parse('')).toBeNull()
  })
})

describe('digitsOrNull', () => {
  it('returns a run of digits, trimmed', () => {
    expect(digitsOrNull(' 32768 ')).toBe('32768')
    expect(digitsOrNull('0')).toBe('0')
  })

  it('returns null for anything else, including an empty field', () => {
    expect(digitsOrNull('')).toBeNull()
    expect(digitsOrNull('12a')).toBeNull()
    expect(digitsOrNull('-1')).toBeNull()
    expect(digitsOrNull(null)).toBeNull()
  })
})

describe('decimalOrNull', () => {
  it('returns a decimal, trimmed', () => {
    expect(decimalOrNull(' 0.4 ')).toBe('0.4')
    expect(decimalOrNull('2')).toBe('2')
  })

  it('returns null for a sign, a malformed number, or an absent parameter', () => {
    expect(decimalOrNull('-0.4')).toBeNull()
    expect(decimalOrNull('.5')).toBeNull()
    expect(decimalOrNull('1.2.3')).toBeNull()
    expect(decimalOrNull(null)).toBeNull()
  })
})

describe('booleanFlag', () => {
  it('reads one and zero as booleans', () => {
    expect(booleanFlag('1')).toBe(true)
    expect(booleanFlag('0')).toBe(false)
  })

  it('returns null for anything else', () => {
    expect(booleanFlag('true')).toBeNull()
    expect(booleanFlag('')).toBeNull()
    expect(booleanFlag(null)).toBeNull()
  })
})

describe('parsePositiveInteger', () => {
  it('reads a whole number of one or more', () => {
    expect(parsePositiveInteger('1')).toBe(1)
    expect(parsePositiveInteger(' 32768 ')).toBe(32768)
  })

  it('returns null for zero, a sign, a fraction or text', () => {
    expect(parsePositiveInteger('0')).toBeNull()
    expect(parsePositiveInteger('-5')).toBeNull()
    expect(parsePositiveInteger('1.5')).toBeNull()
    expect(parsePositiveInteger('many')).toBeNull()
    expect(parsePositiveInteger('')).toBeNull()
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
  provider: { param: 'provider', default: 'huggingface', parse: enumOf(['huggingface', 'modelscope']) },
  context: { param: 'context', default: '32768', parse: digitsOrNull },
  secret: { param: 'secret', default: '', parse: () => null, omit: true },
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
