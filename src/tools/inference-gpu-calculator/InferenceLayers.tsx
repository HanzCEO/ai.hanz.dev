import type { ReactNode } from 'react'

import StepSection from '@/components/layout/StepSection'

/**
 * The two steps of the inference answer.
 *
 * Step one sizes the KV cache, which fixes the model, the context, the
 * sequences and the cache dtypes. Step two turns that into a GPU
 * configuration. The component is presentational: it takes the resolved state
 * and the two forms as children, so it can be rendered in any state without a
 * fetch.
 */

interface InferenceLayersProps {
  cacheOpen: boolean
  gpuOpen: boolean
  onCacheOpenChange: (open: boolean) => void
  onGpuOpenChange: (open: boolean) => void
  /** Step one: the model source, the context, the sequences and the dtypes. */
  cacheForm: ReactNode
  /** Step one's own result, shown under its form. */
  cacheResults: ReactNode
  /** What step one decided, shown once it is folded away. */
  cacheSummary: ReactNode
  /** Step two: the precision, the headroom and the card limit. */
  gpuForm: ReactNode
  /** Step two's answer. */
  gpuResults: ReactNode
  /** True once the cache has a size, which is what unblocks step two. */
  gpuReady: boolean
}

export default function InferenceLayers({
  cacheOpen,
  gpuOpen,
  onCacheOpenChange,
  onGpuOpenChange,
  cacheForm,
  cacheResults,
  cacheSummary,
  gpuForm,
  gpuResults,
  gpuReady,
}: InferenceLayersProps) {
  return (
    <div className="flex flex-col gap-10">
      <StepSection
        step={1}
        title="KV cache"
        description="Choose the model, then set the context length and the sequences you serve. These values set the cache size."
        open={cacheOpen}
        onOpenChange={onCacheOpenChange}
        collapsedSummary={cacheSummary}
      >
        <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
          {cacheForm}
          {cacheResults}
        </div>
      </StepSection>

      <StepSection
        step={2}
        title="GPU configuration"
        description="Set the precision and the memory you hold back. The calculator then tests every card at every card count."
        open={gpuOpen}
        onOpenChange={onGpuOpenChange}
        disabled={!gpuReady}
        collapsedSummary="Size the KV cache in step 1 to see the GPU configuration this model needs."
      >
        <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
          {gpuForm}
          {gpuResults}
        </div>
      </StepSection>
    </div>
  )
}
