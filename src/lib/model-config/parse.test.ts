import { describe, expect, it } from 'vitest'

import { ConfigParseError, parseConfigText } from './parse'

describe('parseConfigText', () => {
  it('accepts a valid config object', () => {
    const config = parseConfigText('{"model_type":"qwen3","num_hidden_layers":36}')
    expect(config.model_type).toBe('qwen3')
    expect(config.num_hidden_layers).toBe(36)
  })

  it('tolerates surrounding whitespace, as a pasted file often carries', () => {
    expect(parseConfigText('\n  {"a":1}\n').a).toBe(1)
  })

  it('rejects empty text', () => {
    expect(() => parseConfigText('   ')).toThrow(ConfigParseError)
    expect(() => parseConfigText('   ')).toThrow(/Paste a config/)
  })

  it('rejects malformed JSON with a message about JSON', () => {
    expect(() => parseConfigText('{bad')).toThrow(ConfigParseError)
    expect(() => parseConfigText('{bad')).toThrow(/not valid JSON/)
  })

  it('rejects a JSON array, since a config is an object', () => {
    expect(() => parseConfigText('[]')).toThrow(ConfigParseError)
    expect(() => parseConfigText('[]')).toThrow(/JSON object/)
  })

  it('rejects a bare primitive', () => {
    expect(() => parseConfigText('42')).toThrow(ConfigParseError)
    expect(() => parseConfigText('"text"')).toThrow(ConfigParseError)
    expect(() => parseConfigText('null')).toThrow(ConfigParseError)
  })

  it('rejects an empty object, which carries no shape to read', () => {
    expect(() => parseConfigText('{}')).toThrow(/empty/)
  })

  it('names the error so a caller can branch on it', () => {
    try {
      parseConfigText('{bad')
      throw new Error('expected a throw')
    } catch (error) {
      expect(error).toBeInstanceOf(ConfigParseError)
      expect((error as ConfigParseError).name).toBe('ConfigParseError')
    }
  })
})
