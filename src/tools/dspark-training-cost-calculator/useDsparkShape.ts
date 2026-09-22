import { useMemo } from 'react'

import { detectDsparkShape, type DsparkTargetShape } from '@/lib/dspark'
import { parseConfigText, type RawConfig } from '@/lib/model-config'
import { useModelConfig } from '@/lib/use-model-config'
import { detectWeightQuantization, type WeightFormatId } from '@/lib/weight-format'

import type { DsparkFormInputs } from './useDsparkState'

export type DsparkShapeStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface DsparkShapeState {
  status: DsparkShapeStatus
  shape: DsparkTargetShape | null
  config: RawConfig | null
  configUrl: string | null
  error: string | null
  /** The weight format the target checkpoint declares, when it declares one. */
  suggestedWeightFormat: WeightFormatId | null
}

const IDLE: DsparkShapeState = {
  status: 'idle',
  shape: null,
  config: null,
  configUrl: null,
  error: null,
  suggestedWeightFormat: null,
}

function numberFrom(value: string): number | null {
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

function optionalFrom(value: string): number | undefined {
  return numberFrom(value) ?? undefined
}

/**
 * Turns the manual fields into a config shape.
 *
 * The manual mode exists for a private or unreleased model. Rather than a
 * second set of formulas, the numbers are written into a config and read back
 * by the same detector every other mode uses, so all three modes agree.
 *
 * An expert bank is only described when a routed expert count is given, which
 * is what keeps a dense target off the mixture of experts path.
 */
export function manualConfig(inputs: DsparkFormInputs): RawConfig | null {
  const hiddenSize = numberFrom(inputs.hiddenSize)
  const numLayers = numberFrom(inputs.numLayers)
  if (!hiddenSize || !numLayers) return null

  const config: RawConfig = {
    model_type: 'manual',
    hidden_size: hiddenSize,
    num_hidden_layers: numLayers,
  }

  const intermediateSize = optionalFrom(inputs.intermediateSize)
  if (intermediateSize) config.intermediate_size = intermediateSize
  const vocabSize = optionalFrom(inputs.vocabSize)
  if (vocabSize) config.vocab_size = vocabSize
  const attentionHeads = optionalFrom(inputs.attentionHeads)
  if (attentionHeads) config.num_attention_heads = attentionHeads
  const headDim = optionalFrom(inputs.headDim)
  if (headDim) config.head_dim = headDim
  const kvHeads = optionalFrom(inputs.kvHeads)
  if (kvHeads) config.num_key_value_heads = kvHeads

  const routedExperts = numberFrom(inputs.routedExperts)
  if (routedExperts) {
    config.n_routed_experts = routedExperts
    const expertsPerToken = optionalFrom(inputs.expertsPerToken)
    if (expertsPerToken) config.num_experts_per_tok = expertsPerToken
    const expertWidth = optionalFrom(inputs.moeIntermediateSize)
    if (expertWidth) config.moe_intermediate_size = expertWidth
    const moeLayers = numberFrom(inputs.moeLayers)
    if (moeLayers) {
      // The detector reads the expert block count as the layers that are not
      // dense, so the dense count is what has to be recorded here.
      config.first_k_dense_replace = Math.max(0, numLayers - moeLayers)
    }
  }

  return config
}

function readyFrom(config: RawConfig, configUrl: string | null): DsparkShapeState {
  const shape = detectDsparkShape(config)
  const quant = detectWeightQuantization(config)
  return {
    status: shape ? 'ready' : 'error',
    shape,
    config,
    configUrl,
    // A config that names only a torch dtype or nothing at all is left to the
    // default, because the picker already opens on BF16 in that case.
    suggestedWeightFormat:
      quant.source === 'assumed' || quant.source === 'torch_dtype' ? null : quant.primary,
    error: shape
      ? null
      : 'That config describes no decoder with a hidden size and a depth. The calculator therefore has nothing to cost.',
  }
}

/**
 * Resolves the target shape from whichever input mode is active.
 *
 * The hub hook is always called so the hook order is stable, and its result is
 * ignored when another mode is selected.
 */
export function useDsparkShape(inputs: DsparkFormInputs): DsparkShapeState {
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
