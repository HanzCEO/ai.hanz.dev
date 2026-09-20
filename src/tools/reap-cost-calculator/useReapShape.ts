import { useMemo } from 'react'

import {
  parseConfigText,
  readObject,
  readString,
  unwrapConfig,
  type RawConfig,
} from '@/lib/model-config'
import { detectMoeShape, type MoeShape, type WeightDtype } from '@/lib/reap'
import { useModelConfig } from '@/lib/use-model-config'

import type { ReapFormInputs } from './useReapState'

export type ReapShapeStatus = 'idle' | 'loading' | 'ready' | 'not-moe' | 'error'

export interface ReapShapeState {
  status: ReapShapeStatus
  shape: MoeShape | null
  config: RawConfig | null
  configUrl: string | null
  error: string | null
  /** The weight format the checkpoint itself declares, when it declares one. */
  suggestedDtype: WeightDtype | null
}

const IDLE: ReapShapeState = {
  status: 'idle',
  shape: null,
  config: null,
  configUrl: null,
  error: null,
  suggestedDtype: null,
}

function numberFrom(value: string): number | null {
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

/**
 * Turns the manual fields into a config shape.
 *
 * The manual mode exists for a private or unreleased model. Rather than a
 * second set of formulas, the numbers are written into a config and read back
 * by the same detector every other mode uses, so all three modes agree.
 */
export function manualConfig(inputs: ReapFormInputs): RawConfig | null {
  const hiddenSize = numberFrom(inputs.hiddenSize)
  const expertWidth = numberFrom(inputs.moeIntermediateSize)
  const routedExperts = numberFrom(inputs.routedExperts)
  const expertsPerToken = numberFrom(inputs.expertsPerToken)
  const numLayers = numberFrom(inputs.numLayers)
  const moeLayers = numberFrom(inputs.moeLayers)

  if (!hiddenSize || !expertWidth || !routedExperts || !expertsPerToken || !numLayers || !moeLayers) {
    return null
  }

  const config: RawConfig = {
    model_type: 'manual',
    hidden_size: hiddenSize,
    moe_intermediate_size: expertWidth,
    n_routed_experts: routedExperts,
    num_experts_per_tok: expertsPerToken,
    num_hidden_layers: numLayers,
    n_shared_experts: numberFrom(inputs.sharedExperts) ?? 0,
    shared_expert_intermediate_size:
      numberFrom(inputs.sharedExpertIntermediate) ?? expertWidth,
    // The detector reads the MoE layer count as the layers that are not dense.
    first_k_dense_replace: Math.max(0, numLayers - moeLayers),
  }

  const denseFfn = numberFrom(inputs.intermediateSize)
  if (denseFfn) config.intermediate_size = denseFfn
  const vocabSize = numberFrom(inputs.vocabSize)
  if (vocabSize) config.vocab_size = vocabSize
  const attentionHeads = numberFrom(inputs.attentionHeads)
  if (attentionHeads) config.num_attention_heads = attentionHeads
  const headDim = numberFrom(inputs.headDim)
  if (headDim) config.head_dim = headDim
  const kvHeads = numberFrom(inputs.kvHeads)
  if (kvHeads) config.num_key_value_heads = kvHeads

  return config
}

/** Reads the precision the checkpoint was published in, if it says. */
function suggestedDtypeFor(config: RawConfig): WeightDtype | null {
  const { inner, outer } = unwrapConfig(config)
  const quant = readObject(outer, 'quantization_config') ?? readObject(inner, 'quantization_config')
  if (!quant) return null

  const method = (readString(quant, 'quant_method') ?? '').toLowerCase()
  const format = (readString(quant, 'fmt') ?? '').toLowerCase()

  if (method.includes('fp8') || format.includes('e4m3') || format.includes('e5m2')) return 'FP8'
  if (
    method.includes('fp4') ||
    method.includes('nvfp4') ||
    method.includes('mxfp4') ||
    format.includes('e2m1')
  ) {
    return 'INT4'
  }
  return null
}

function readyFrom(config: RawConfig, configUrl: string | null): ReapShapeState {
  const shape = detectMoeShape(config)
  return {
    status: shape ? 'ready' : 'not-moe',
    shape,
    config,
    configUrl,
    error: null,
    suggestedDtype: suggestedDtypeFor(config),
  }
}

/**
 * Resolves the model shape from whichever input mode is active.
 *
 * The hub hook is always called so the hook order is stable, and its result is
 * ignored when another mode is selected.
 */
export function useReapShape(inputs: ReapFormInputs): ReapShapeState {
  const hub = useModelConfig(inputs.provider, inputs.modelId, inputs.token)

  const pasted = useMemo(() => {
    if (inputs.mode !== 'paste') return null
    if (inputs.configText.trim() === '') return null
    try {
      return { config: parseConfigText(inputs.configText), error: null }
    } catch (error) {
      return {
        config: null,
        error: error instanceof Error ? error.message : 'The calculator cannot read that config.',
      }
    }
  }, [inputs.mode, inputs.configText])

  const manual = useMemo(() => {
    if (inputs.mode !== 'manual') return null
    return manualConfig(inputs)
  }, [inputs])

  if (inputs.mode === 'hub') {
    if (hub.status === 'idle') return IDLE
    if (hub.status === 'loading') return { ...IDLE, status: 'loading' }
    if (hub.status === 'error') {
      return {
        ...IDLE,
        status: 'error',
        error: hub.error?.message ?? 'The calculator cannot read that model config.',
      }
    }
    if (!hub.config) return IDLE
    return readyFrom(hub.config, hub.url)
  }

  if (inputs.mode === 'paste') {
    if (!pasted) return IDLE
    if (!pasted.config) return { ...IDLE, status: 'error', error: pasted.error }
    return readyFrom(pasted.config, null)
  }

  if (!manual) return IDLE
  return readyFrom(manual, null)
}
