import { describe, expect, it } from 'vitest'

import { MANUAL_DEFAULTS, manualConfig } from './manual'

describe('manualConfig', () => {
  it('describes a dense model with no expert fields', () => {
    const config = manualConfig(MANUAL_DEFAULTS)
    expect(config).not.toBeNull()
    expect(config?.hidden_size).toBe(4096)
    expect(config?.num_hidden_layers).toBe(36)
    expect(config?.intermediate_size).toBe(12288)
    expect(config?.vocab_size).toBe(151936)
    expect(config?.num_attention_heads).toBe(32)
    expect(config?.num_key_value_heads).toBe(8)
    expect(config?.head_dim).toBe(128)
    expect(config).not.toHaveProperty('n_routed_experts')
    expect(config).not.toHaveProperty('moe_intermediate_size')
    expect(config).not.toHaveProperty('first_k_dense_replace')
  })

  it('returns null when the hidden size or the layer count is missing', () => {
    expect(manualConfig({ ...MANUAL_DEFAULTS, hiddenSize: '' })).toBeNull()
    expect(manualConfig({ ...MANUAL_DEFAULTS, hiddenSize: '0' })).toBeNull()
    expect(manualConfig({ ...MANUAL_DEFAULTS, numLayers: '' })).toBeNull()
    expect(manualConfig({ ...MANUAL_DEFAULTS, numLayers: '0' })).toBeNull()
  })

  it('returns null for a fraction, a sign or text', () => {
    expect(manualConfig({ ...MANUAL_DEFAULTS, hiddenSize: '4096.5' })).toBeNull()
    expect(manualConfig({ ...MANUAL_DEFAULTS, numLayers: '-36' })).toBeNull()
    expect(manualConfig({ ...MANUAL_DEFAULTS, numLayers: 'many' })).toBeNull()
  })

  it('adds the expert bank once a routed expert count is given', () => {
    const config = manualConfig({
      ...MANUAL_DEFAULTS,
      routedExperts: '64',
      expertsPerToken: '6',
      moeIntermediateSize: '768',
      moeLayers: '21',
    })
    expect(config?.n_routed_experts).toBe(64)
    expect(config?.num_experts_per_tok).toBe(6)
    expect(config?.moe_intermediate_size).toBe(768)
    // The detector reads the expert count as the layers that are not dense, so
    // the dense count is what has to be recorded.
    expect(config?.first_k_dense_replace).toBe(36 - 21)
  })

  it('never records a negative dense layer count', () => {
    const config = manualConfig({
      ...MANUAL_DEFAULTS,
      numLayers: '10',
      routedExperts: '8',
      moeLayers: '40',
    })
    expect(config?.first_k_dense_replace).toBe(0)
  })

  it('records a tied head as true and a separate head as false', () => {
    expect(manualConfig({ ...MANUAL_DEFAULTS, tieEmbeddings: 'tied' })?.tie_word_embeddings).toBe(
      true,
    )
    expect(
      manualConfig({ ...MANUAL_DEFAULTS, tieEmbeddings: 'untied' })?.tie_word_embeddings,
    ).toBe(false)
  })

  it('treats a zero in an optional field as absent', () => {
    const config = manualConfig({
      ...MANUAL_DEFAULTS,
      vocabSize: '0',
      attentionHeads: '0',
      kvHeads: '0',
      headDim: '0',
    })
    expect(config).not.toHaveProperty('vocab_size')
    expect(config).not.toHaveProperty('num_attention_heads')
    expect(config).not.toHaveProperty('num_key_value_heads')
    expect(config).not.toHaveProperty('head_dim')
  })
})
