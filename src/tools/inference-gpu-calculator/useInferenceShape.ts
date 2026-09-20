import { useMemo } from 'react'

import { detectModelShape, type ModelShape } from '@/lib/model-shape'
import { manualConfig, parseConfigText, type RawConfig } from '@/lib/model-config'
import { useModelConfig } from '@/lib/use-model-config'

import type { InferenceFormInputs } from './useInferenceState'

export type InferenceShapeStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface InferenceShapeState {
  status: InferenceShapeStatus
  shape: ModelShape | null
  config: RawConfig | null
  configUrl: string | null
  error: string | null
}

const IDLE: InferenceShapeState = {
  status: 'idle',
  shape: null,
  config: null,
  configUrl: null,
  error: null,
}

function readyFrom(config: RawConfig, configUrl: string | null): InferenceShapeState {
  const shape = detectModelShape(config)
  return {
    status: shape ? 'ready' : 'error',
    shape,
    config,
    configUrl,
    error: shape
      ? null
      : 'That config describes no transformer with a hidden size and a depth. The calculator therefore has nothing to size.',
  }
}

/**
 * Resolves the model shape from whichever input mode is active.
 *
 * The hub hook is always called so the hook order is stable, and its result is
 * ignored when another mode is selected.
 */
export function useInferenceShape(inputs: InferenceFormInputs): InferenceShapeState {
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
