import { useMemo } from 'react'

import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import {
  INFERENCE_FAQ,
  InferenceInputError,
  estimateInference,
  type InferenceInputField,
  type InferenceInputs,
  type InferenceResult,
} from '@/lib/inference'
import { parsePositiveInteger } from '@/lib/url-state'

import InferenceForm from '@/tools/inference-gpu-calculator/InferenceForm'
import InferenceResults from '@/tools/inference-gpu-calculator/InferenceResults'
import { useInferenceShape } from '@/tools/inference-gpu-calculator/useInferenceShape'
import { useInferenceState } from '@/tools/inference-gpu-calculator/useInferenceState'

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
  const { inputs, update } = useInferenceState()
  const shapeState = useInferenceShape(inputs)

  const contextLength = parsePositiveInteger(inputs.contextLength)
  const sequences = parsePositiveInteger(inputs.sequences)
  const maxGpus = parsePositiveInteger(inputs.maxGpus)
  const headroomPercent = parsePercent(inputs.headroomPercent)

  const contextError =
    contextLength === null ? 'Enter a whole number of tokens, 1 or more.' : null
  const sequenceError = sequences === null ? 'Enter a whole number, 1 or more.' : null
  const maxGpusError = maxGpus === null ? 'Enter a whole number, 1 or more.' : null
  const headroomError =
    headroomPercent === null ? 'Enter a percentage from 0 to 99.' : null

  const config = shapeState.status === 'ready' ? shapeState.config : null
  const shape = shapeState.status === 'ready' ? shapeState.shape : null

  const { result, computeError, invalidField } = useMemo((): {
    result: InferenceResult | null
    computeError: string | null
    invalidField: InferenceInputField | null
  } => {
    if (!config || !shape) return { result: null, computeError: null, invalidField: null }
    if (
      contextLength === null ||
      sequences === null ||
      maxGpus === null ||
      headroomPercent === null
    ) {
      return { result: null, computeError: null, invalidField: null }
    }

    const inferenceInputs: InferenceInputs = {
      config,
      precision: inputs.precision,
      contextLength,
      sequences,
      headroom: headroomPercent / 100,
      maxGpus,
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
          error instanceof Error
            ? error.message
            : 'The calculator cannot size that model.',
        invalidField: error instanceof InferenceInputError ? error.field : null,
      }
    }
  }, [config, shape, contextLength, sequences, maxGpus, headroomPercent, inputs.precision])

  // The name the reader used for the model, so the headline sentence names the
  // model they typed rather than the model_type inside the config.
  const modelLabel = inputs.mode === 'hub' ? inputs.modelId : shape?.modelType

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="Inference GPU Calculator" />

      <ToolHeader
        title="Inference GPU Calculator"
        description="Find the GPU configuration that holds a model in FP16 or BF16. The calculator adds the resident weights, the KV cache, the activation buffer and the framework reserve. It then tests every card in its list at every card count, and reports the smallest configuration that fits. The model shape comes from a HuggingFace or ModelScope config. You can also paste a config.json or enter the values yourself."
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <InferenceForm
          inputs={inputs}
          update={update}
          shapeState={shapeState}
          invalidField={invalidField}
          contextError={contextError}
          sequenceError={sequenceError}
          headroomError={headroomError}
          maxGpusError={maxGpusError}
        />
        <InferenceResults
          result={result}
          shapeState={shapeState}
          computeError={computeError}
          modelLabel={modelLabel}
        />
      </div>

      <FaqSection
        id="inference-faq-heading"
        heading="Questions about inference hardware"
        items={INFERENCE_FAQ}
      />
    </div>
  )
}
