import { describe, expect, it } from 'vitest'

import {
  dtypeNameToBytes,
  readArrayFrom,
  readBoolean,
  readFlagArray,
  readNumber,
  readNumberArray,
  readNumberFrom,
  readObject,
  readString,
  readStringArray,
  readStringFrom,
  unwrapConfig,
} from './readers'
import type { RawConfig } from './types'

describe('readNumber', () => {
  it('reads a finite number', () => {
    expect(readNumber({ layers: 36 }, 'layers')).toBe(36)
  })

  it('reads a numeric string, since hubs return both forms', () => {
    expect(readNumber({ layers: '36' }, 'layers')).toBe(36)
  })

  it('rejects a non-numeric string, an empty string, and a non-finite number', () => {
    expect(readNumber({ layers: 'many' }, 'layers')).toBeUndefined()
    expect(readNumber({ layers: '   ' }, 'layers')).toBeUndefined()
    expect(readNumber({ layers: Number.POSITIVE_INFINITY }, 'layers')).toBeUndefined()
  })

  it('returns undefined for a missing key or a wrong type', () => {
    expect(readNumber({}, 'layers')).toBeUndefined()
    expect(readNumber({ layers: true }, 'layers')).toBeUndefined()
  })
})

describe('readNumberArray and readStringArray', () => {
  it('returns a homogeneous array', () => {
    expect(readNumberArray({ ratios: [2, 4] }, 'ratios')).toEqual([2, 4])
    expect(readStringArray({ types: ['full', 'shared'] }, 'types')).toEqual(['full', 'shared'])
  })

  it('rejects a mixed array and an empty array', () => {
    expect(readNumberArray({ ratios: [2, 'four'] }, 'ratios')).toBeUndefined()
    expect(readNumberArray({ ratios: [] }, 'ratios')).toBeUndefined()
    expect(readStringArray({ types: [] }, 'types')).toBeUndefined()
  })
})

describe('readFlagArray', () => {
  it('reads a boolean array as it stands', () => {
    expect(readFlagArray({ pattern: [true, false] }, 'pattern')).toEqual([true, false])
  })

  it('reads a zero and one array, which is how MiMo-V2.6 writes it', () => {
    expect(readFlagArray({ moe_layer_freq: [0, 1, 1] }, 'moe_layer_freq')).toEqual([
      false,
      true,
      true,
    ])
  })

  it('rejects a scalar, an empty array, a mixed array, and null', () => {
    expect(readFlagArray({ freq: 1 }, 'freq')).toBeUndefined()
    expect(readFlagArray({ freq: [] }, 'freq')).toBeUndefined()
    expect(readFlagArray({ freq: [0, true] }, 'freq')).toBeUndefined()
    expect(readFlagArray({ freq: null }, 'freq')).toBeUndefined()
    expect(readFlagArray({}, 'freq')).toBeUndefined()
  })

  it('treats a non-finite number entry as unusable rather than truthy', () => {
    expect(readFlagArray({ freq: [0, Number.NaN] }, 'freq')).toBeUndefined()
  })
})

describe('readString, readBoolean, readObject', () => {
  it('reads the declared type only', () => {
    expect(readString({ model_type: 'qwen3' }, 'model_type')).toBe('qwen3')
    expect(readString({ model_type: 3 }, 'model_type')).toBeUndefined()
    expect(readBoolean({ tie: false }, 'tie')).toBe(false)
    expect(readBoolean({ tie: 'false' }, 'tie')).toBeUndefined()
  })

  it('reads a nested object but not an array', () => {
    expect(readObject({ cfg: { a: 1 } }, 'cfg')).toEqual({ a: 1 })
    expect(readObject({ cfg: [1] }, 'cfg')).toBeUndefined()
    expect(readObject({ cfg: null }, 'cfg')).toBeUndefined()
  })
})

describe('readNumberFrom and readStringFrom and readArrayFrom', () => {
  it('returns the first present key in the order given', () => {
    const config: RawConfig = { num_experts: 512, n_routed_experts: 256 }
    expect(readNumberFrom(config, ['n_routed_experts', 'num_experts'])).toBe(256)
    expect(readNumberFrom(config, ['missing', 'num_experts'])).toBe(512)
    expect(readNumberFrom(config, ['missing'])).toBeUndefined()
  })

  it('picks a string by the same rule', () => {
    const config: RawConfig = { dtype: 'bfloat16' }
    expect(readStringFrom(config, ['torch_dtype', 'dtype'])).toBe('bfloat16')
  })

  it('picks an array by kind', () => {
    const config: RawConfig = { a: ['x'], b: [1, 2] }
    expect(readArrayFrom(config, ['b'], 'number')).toEqual([1, 2])
    expect(readArrayFrom(config, ['a'], 'string')).toEqual(['x'])
    expect(readArrayFrom(config, ['a'], 'number')).toBeUndefined()
  })
})

describe('unwrapConfig', () => {
  it('resolves a text_config nested language model', () => {
    const raw: RawConfig = {
      model_type: 'gemma4',
      text_config: { model_type: 'gemma4_text', num_hidden_layers: 60 },
    }
    const { inner, outer } = unwrapConfig(raw)
    expect(inner.model_type).toBe('gemma4_text')
    expect(inner.num_hidden_layers).toBe(60)
    // The outer config is returned unchanged so outer-only fields stay reachable.
    expect(outer).toBe(raw)
  })

  it('resolves llm_config and language_config too', () => {
    expect(unwrapConfig({ llm_config: { a: 1 } }).inner).toEqual({ a: 1 })
    expect(unwrapConfig({ language_config: { a: 2 } }).inner).toEqual({ a: 2 })
  })

  it('returns the config itself when nothing is nested', () => {
    const raw: RawConfig = { model_type: 'qwen3' }
    const { inner, outer } = unwrapConfig(raw)
    expect(inner).toBe(raw)
    expect(outer).toBe(raw)
  })

  it('ignores an array under a nested key', () => {
    const raw: RawConfig = { text_config: [1, 2] }
    expect(unwrapConfig(raw).inner).toBe(raw)
  })
})

describe('dtypeNameToBytes', () => {
  it('maps the torch names hubs actually publish', () => {
    expect(dtypeNameToBytes('bfloat16')).toBe(2)
    expect(dtypeNameToBytes('torch.float16')).toBe(2)
    expect(dtypeNameToBytes('float32')).toBe(4)
    expect(dtypeNameToBytes('fp8')).toBe(1)
    expect(dtypeNameToBytes('int8')).toBe(1)
  })

  it('returns undefined for an unknown or absent name', () => {
    expect(dtypeNameToBytes('mystery')).toBeUndefined()
    expect(dtypeNameToBytes(undefined)).toBeUndefined()
  })
})
