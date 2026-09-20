import { useMemo } from 'react'

import {
  INFERENCE_FAQ,
  InferenceInputError,
  estimateInference,
  type InferenceInputField,
  type InferenceInputs,
  type InferenceResult,
} from '@/lib/inference'
import { detectModelShape, type ModelShape } from '@/lib/model-shape'
import { useConfigSource } from '@/lib/use-config-source'
import { parsePositiveInteger } from '@/lib/url-state'
import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import CalculatorForm from '@/tools/kv-cache-calculator/CalculatorForm'
import CalculatorResults from '@/tools/kv-cache-calculator/CalculatorResults'
import { useKvCacheLayer } from '@/tools/kv-cache-calculator/useKvCacheLayer'

import { describeCacheStep } from '@/tools/inference-gpu-calculator/cacheSummary'
import InferenceForm from '@/tools/inference-gpu-calculator/InferenceForm'
import InferenceLayers from '@/tools/inference-gpu-calculator/InferenceLayers'
import InferenceResults from '@/tools/inference-gpu-calculator/InferenceResults'
import { useInferenceState } from '@/tools/inference-gpu-calculator/useInferenceState'
import { useLayers } from '@/tools/inference-gpu-calculator/useLayers'

/**
 * Reads the VRAM headroom, which is held as a percentage.
 *
 * The engine takes a fraction, so the field is divided by 100. A value of 100
 * or more is rejected rather than clamped, because it would mean the card holds
 * nothing at all.
 */
function parsePercent(value: string): number | null {
  const trimmed = value.trim()
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed) || parsed < 0 || parsed >= 100) return null
  return parsed
}

export default function InferenceGpuCalculator() {
  const { inputs, update, seeded } = useInferenceState()
  const configState = useConfigSource(inputs)
  const cache = useKvCacheLayer({ inputs, update, configState, seeded })

  const config = configState.status === 'ready' ? configState.config : null

  const shape: ModelShape | null = useMemo(
    () => (config ? detectModelShape(config) : null),
    [config],
  )

  const headroomPercent = parsePercent(inputs.headroomPercent)
  const maxGpus = parsePositiveInteger(inputs.maxGpus)

  const headroomError = headroomPercent === null ? 'Enter a percentage from 0 to 99.' : null
  const maxGpusError = maxGpus === null ? 'Enter a whole number, 1 or more.' : null

  const { result, computeError, invalidField } = useMemo((): {
    result: InferenceResult | null
    computeError: string | null
    invalidField: InferenceInputField | null
  } => {
    const contextLength = parsePositiveInteger(inputs.contextLength)
    const sequenceCount = parsePositiveInteger(inputs.sequenceCount)
    if (!config || !shape) return { result: null, computeError: null, invalidField: null }
    if (contextLength === null || sequenceCount === null) {
      return { result: null, computeError: null, invalidField: null }
    }
    if (headroomPercent === null || maxGpus === null) {
      return { result: null, computeError: null, invalidField: null }
    }

    const inferenceInputs: InferenceInputs = {
      config,
      precision: inputs.precision,
      contextLength,
      sequences: sequenceCount,
      headroom: headroomPercent / 100,
      maxGpus,
      kvCacheDtype: inputs.kvCacheDtype,
      indexerDtype: inputs.indexerDtype,
    }

    try {
      return {
        result: estimateInference(shape, inferenceInputs),
        computeError: null,
        invalidField: null,
      }
    } catch (error) {
      return {
        result: null,
        computeError:
          error instanceof Error ? error.message : 'The calculator cannot size that model.',
        invalidField: error instanceof InferenceInputError ? error.field : null,
      }
    }
  }, [
    config,
    shape,
    inputs.contextLength,
    inputs.sequenceCount,
    inputs.precision,
    inputs.kvCacheDtype,
    inputs.indexerDtype,
    headroomPercent,
    maxGpus,
  ])

  const cacheResult = cache.result
  const { cacheOpen, gpuOpen, setCacheOpen, setGpuOpen } = useLayers(cacheResult !== null)

  // The name the reader used for the model, so the headline sentence names the
  // model they typed rather than the model_type inside the config.
  const modelLabel = inputs.mode === 'hub' ? inputs.modelId : shape?.modelType

  const cacheSummary = cacheResult
    ? describeCacheStep(
        cacheResult,
        inputs.modelId,
        inputs.kvCacheDtype,
        inputs.indexerDtype,
      )
    : 'Enter a model, a context length, and a sequence count to size the cache.'

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="Inference GPU Calculator" />

      <ToolHeader
        title="Inference GPU Calculator"
        description="Answer the hardware question in 2 steps. Step 1 sizes the KV cache from the model, the context length, and the sequences you serve. Step 2 turns that into the smallest GPU configuration that holds the run in FP16 or BF16, and reports the decode rate it reaches. The model shape comes from a HuggingFace or ModelScope config, from a config.json you paste, or from values you enter yourself."
      />

      <InferenceLayers
        cacheOpen={cacheOpen}
        gpuOpen={gpuOpen}
        onCacheOpenChange={setCacheOpen}
        onGpuOpenChange={setGpuOpen}
        gpuReady={cacheResult !== null}
        cacheSummary={cacheSummary}
        cacheForm={
          <CalculatorForm
            inputs={inputs}
            update={update}
            configState={configState}
            result={cacheResult}
            contextError={cache.contextError}
            sequenceError={cache.sequenceError}
            onUseMaxContext={cache.onUseMaxContext}
          />
        }
        cacheResults={
          <CalculatorResults
            result={cacheResult}
            configState={configState}
            onSwitchProvider={cache.onSwitchProvider}
            computeError={cache.computeError}
          />
        }
        gpuForm={
          <InferenceForm
            inputs={inputs}
            update={update}
            invalidField={invalidField}
            headroomError={headroomError}
            maxGpusError={maxGpusError}
          />
        }
        gpuResults={
          <InferenceResults
            result={result}
            configState={configState}
            shape={shape}
            computeError={computeError}
            modelLabel={modelLabel}
          />
        }
      />

      <FaqSection
        id="inference-faq-heading"
        heading="Questions about inference hardware"
        items={INFERENCE_FAQ}
      />
    </div>
  )
}
