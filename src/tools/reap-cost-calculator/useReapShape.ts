import { useMemo } from 'react'

import {
  parseConfigText,
  type RawConfig,
} from '@/lib/model-config'
import { detectMoeShape, type MoeShape } from '@/lib/reap'
import { useModelConfig } from '@/lib/use-model-config'
import { detectWeightQuantization, type WeightFormatId } from '@/lib/weight-format'

import type { ReapFormInputs } from './useReapState'

export type ReapShapeStatus = 'idle' | 'loading' | 'ready' | 'not-moe' | 'error'

export interface ReapShapeState {
  status: ReapShapeStatus
  shape: MoeShape | null
  config: RawConfig | null
  configUrl: string | null
  error: string | null
  /** The weight format the checkpoint itself declares, when it declares one. */
  suggestedDtype: WeightFormatId | null
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

/**
 * Reads the weight format the checkpoint was published in, if it says.
 *
 * The calibration reads the whole checkpoint, but the VRAM peak is one expert
 * block, so the expert bucket is the format that decides the fit. A config that
 * names only a torch dtype or nothing at all is left to the default, because
 * the panel already opens on BF16 in that case.
 */
export function suggestedDtypeFor(config: RawConfig): WeightFormatId | null {
  const quant = detectWeightQuantization(config)
  if (quant.source === 'assumed' || quant.source === 'torch_dtype') return null
  return quant.experts
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
