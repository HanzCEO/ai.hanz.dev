import { useEffect, useMemo, useRef } from 'react'

import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import {
  DEFAULT_GPU_ID,
  DEFAULT_STORAGE_ID,
  findGpu,
  findStorage,
  type GpuSpec,
  type StorageSpec,
} from '@/lib/hardware'
import {
  DEFAULT_SETUP_SECONDS,
  REAP_FAQ,
  ReapInputError,
  estimateReap,
  type ReapInputField,
  type ReapInputs,
  type ReapResult,
} from '@/lib/reap'
import ReapForm from '@/tools/reap-cost-calculator/ReapForm'
import ReapResults from '@/tools/reap-cost-calculator/ReapResults'
import { useReapShape } from '@/tools/reap-cost-calculator/useReapShape'
import { useReapState } from '@/tools/reap-cost-calculator/useReapState'

export default function ReapCostCalculator() {
  const { inputs, update, seeded } = useReapState()
  const shapeState = useReapShape(inputs)

  const gpu: GpuSpec = findGpu(inputs.gpuId) ?? (findGpu(DEFAULT_GPU_ID) as GpuSpec)
  const storage: StorageSpec =
    findStorage(inputs.storageId) ?? (findStorage(DEFAULT_STORAGE_ID) as StorageSpec)

  const { result, computeError, invalidField } = useMemo((): {
    result: ReapResult | null
    computeError: string | null
    invalidField: ReapInputField | null
  } => {
    if (shapeState.status !== 'ready' || !shapeState.shape) {
      return { result: null, computeError: null, invalidField: null }
    }

    const reapInputs: ReapInputs = {
      calibrationSamples: Number(inputs.samples),
      sequenceLength: Number(inputs.sequenceLength),
      pruneRatio: Number(inputs.pruneRatio),
      gpu,
      storage,
      weightDtype: inputs.weightDtype,
      mfu: Number(inputs.mfu),
      overheadFactor: Number(inputs.overheadFactor),
      setupSeconds: DEFAULT_SETUP_SECONDS,
      microBatchSize: Number(inputs.microBatchSize),
      scaleTopK: inputs.scaleTopK,
    }

    try {
      return {
        result: estimateReap(shapeState.shape, reapInputs),
        computeError: null,
        invalidField: null,
      }
    } catch (error) {
      return {
        result: null,
        computeError:
          error instanceof Error ? error.message : 'Those inputs could not be costed.',
        invalidField: error instanceof ReapInputError ? error.field : null,
      }
    }
  }, [shapeState.status, shapeState.shape, inputs, gpu, storage])

  // Open a newly selected model on the precision its checkpoint ships in. A
  // shared link that names a precision keeps its own choice.
  const dtypeSeededFor = useRef<string | null>(null)
  useEffect(() => {
    if (shapeState.status !== 'ready') return
    const selection = `${inputs.mode}:${inputs.modelId}:${shapeState.shape?.modelType ?? ''}`
    if (dtypeSeededFor.current === selection) return
    dtypeSeededFor.current = selection
    if (seeded.current.dtype) return
    if (shapeState.suggestedDtype) update({ weightDtype: shapeState.suggestedDtype })
  }, [shapeState.status, shapeState.shape, shapeState.suggestedDtype, inputs.mode, inputs.modelId, seeded, update])

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="REAP Cost Calculator" />

      <ToolHeader
        title="REAP Cost Calculator"
        description="This REAP duration calculator estimates the duration of a run of Router-weighted Expert Activation Pruning on your hardware. It also shows whether one expert block fits in your VRAM, and how much smaller the pruned model becomes. The calculator reads the model shape from a HuggingFace or ModelScope config. You can also paste a config.json or enter the values yourself."
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <ReapForm
          inputs={inputs}
          update={update}
          shapeState={shapeState}
          gpu={gpu}
          invalidField={invalidField}
        />
        <ReapResults
          result={result}
          shapeState={shapeState}
          gpu={gpu}
          computeError={computeError}
        />
      </div>

      <FaqSection id="reap-faq-heading" heading="Questions about REAP" items={REAP_FAQ} />
    </div>
  )
}
