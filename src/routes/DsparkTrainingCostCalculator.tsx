import { useMemo } from 'react'

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
  DSPARK_FAQ,
  DsparkInputError,
  estimateDspark,
  type DsparkInputField,
  type DsparkInputs,
  type DsparkResult,
} from '@/lib/dspark'
import DsparkForm from '@/tools/dspark-training-cost-calculator/DsparkForm'
import DsparkResults from '@/tools/dspark-training-cost-calculator/DsparkResults'
import { useDsparkShape } from '@/tools/dspark-training-cost-calculator/useDsparkShape'
import { useDsparkState } from '@/tools/dspark-training-cost-calculator/useDsparkState'

/** Reads a field the form holds as text into a number the estimator can use. */
function count(value: string): number {
  const parsed = Number(value.trim())
  return Number.isFinite(parsed) ? parsed : Number.NaN
}

export default function DsparkTrainingCostCalculator() {
  const { inputs, update } = useDsparkState()
  const shapeState = useDsparkShape(inputs)

  const gpu: GpuSpec = findGpu(inputs.gpuId) ?? (findGpu(DEFAULT_GPU_ID) as GpuSpec)
  const storage: StorageSpec =
    findStorage(inputs.storageId) ?? (findStorage(DEFAULT_STORAGE_ID) as StorageSpec)

  const { result, computeError, invalidField } = useMemo((): {
    result: DsparkResult | null
    computeError: string | null
    invalidField: DsparkInputField | null
  } => {
    if (shapeState.status !== 'ready' || !shapeState.shape) {
      return { result: null, computeError: null, invalidField: null }
    }

    const samples = count(inputs.samples)
    const sequenceLength = count(inputs.sequenceLength)

    const dsparkInputs: DsparkInputs = {
      numTargetLayers: count(inputs.numTargetLayers),
      // The form asks for samples and a packed length, because that is how a
      // dataset is described. The estimator works in tokens, because that is
      // what drives both the cache and the run.
      trainingTokens: samples * sequenceLength,
      epochs: count(inputs.epochs),
      numAnchors: count(inputs.numAnchors),
      blockSize: count(inputs.blockSize),
      numDraftLayers: count(inputs.numDraftLayers),
      markovRank: count(inputs.markovRank),
      sequenceLength,
      dataMode: inputs.dataMode,
      gpu,
      storage,
      gpuCount: count(inputs.gpuCount),
      mfu: count(inputs.mfu),
      overheadFactor: count(inputs.overheadFactor),
      setupSeconds: 600,
      microBatchSize: count(inputs.microBatchSize),
    }

    try {
      return {
        result: estimateDspark(shapeState.shape, dsparkInputs),
        computeError: null,
        invalidField: null,
      }
    } catch (error) {
      return {
        result: null,
        computeError:
          error instanceof Error ? error.message : 'Those inputs could not be costed.',
        invalidField: error instanceof DsparkInputError ? error.field : null,
      }
    }
  }, [shapeState.status, shapeState.shape, inputs, gpu, storage])

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="DSpark Training Cost Calculator" />

      <ToolHeader
        title="DSpark Training Cost Calculator"
        description="Estimate what it costs to train a DSpark speculative-decoding drafter against a custom target. The calculator answers 3 questions. It shows the size of the target cache, the duration of the run, and whether the run fits in VRAM. It reads the target shape from a HuggingFace or ModelScope config. You can also paste a config.json or enter the values yourself."
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <DsparkForm
          inputs={inputs}
          update={update}
          shapeState={shapeState}
          gpu={gpu}
          invalidField={invalidField}
        />
        <DsparkResults
          result={result}
          shapeState={shapeState}
          gpu={gpu}
          computeError={computeError}
        />
      </div>

      <FaqSection id="dspark-faq-heading" heading="Questions about DSpark" items={DSPARK_FAQ} />
    </div>
  )
}
