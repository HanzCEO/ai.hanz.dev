import { useEffect, useMemo, useRef } from 'react'
import { Link } from 'react-router'

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
  estimateReap,
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

  const { result, computeError } = useMemo((): {
    result: ReapResult | null
    computeError: string | null
  } => {
    if (shapeState.status !== 'ready' || !shapeState.shape) {
      return { result: null, computeError: null }
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
      return { result: estimateReap(shapeState.shape, reapInputs), computeError: null }
    } catch (error) {
      return {
        result: null,
        computeError:
          error instanceof Error ? error.message : 'Those inputs could not be costed.',
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
      <nav className="text-muted-foreground text-xs">
        <Link to="/" className="underline-offset-4 hover:underline">
          Tools
        </Link>
        <span aria-hidden="true"> / </span>
        <span>REAP Cost Calculator</span>
      </nav>

      <header className="flex max-w-3xl flex-col gap-3">
        <h1 className="text-2xl font-medium tracking-tight">REAP Cost Calculator</h1>
        <p className="text-muted-foreground">
          This REAP duration calculator estimates how long a Router-weighted Expert Activation
          Pruning run takes on your hardware, whether one expert block fits in your VRAM, and how
          much smaller the pruned model gets. The shape is read from a HuggingFace or ModelScope
          config, a config.json you paste, or numbers you type.
        </p>
      </header>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <ReapForm inputs={inputs} update={update} shapeState={shapeState} gpu={gpu} />
        <ReapResults
          result={result}
          shapeState={shapeState}
          gpu={gpu}
          computeError={computeError}
        />
      </div>

      <section aria-labelledby="reap-faq-heading" className="flex max-w-3xl flex-col gap-5">
        <h2 id="reap-faq-heading" className="text-lg font-medium tracking-tight">
          Questions about REAP
        </h2>
        <div className="flex flex-col gap-5">
          {REAP_FAQ.map((item) => (
            <div key={item.question}>
              <h3 className="text-sm font-medium">{item.question}</h3>
              <p className="text-muted-foreground mt-1 text-sm">{item.answer}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
