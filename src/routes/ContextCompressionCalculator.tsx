import { useMemo } from 'react'

import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import {
  COMPRESSION_FAQ,
  CompressionInputError,
  estimateCompression,
  type CompressionInputField,
  type CompressionInputs,
  type CompressionResult,
} from '@/lib/compression'
import CompressionForm from '@/tools/context-compression-calculator/CompressionForm'
import CompressionResults from '@/tools/context-compression-calculator/CompressionResults'
import { useCompressionState } from '@/tools/context-compression-calculator/useCompressionState'

/** Reads the seven text fields as numbers, so the estimator sees real values. */
function toInputs(fields: {
  inputPrice: string
  cachedInputPrice: string
  outputPrice: string
  missPercent: string
  cachePercent: string
  outputPercent: string
  compressionPercent: string
}): CompressionInputs {
  return {
    inputPrice: Number(fields.inputPrice),
    cachedInputPrice: Number(fields.cachedInputPrice),
    outputPrice: Number(fields.outputPrice),
    mix: {
      missPercent: Number(fields.missPercent),
      cachePercent: Number(fields.cachePercent),
      outputPercent: Number(fields.outputPercent),
    },
    compressionPercent: Number(fields.compressionPercent),
  }
}

export default function ContextCompressionCalculator() {
  const { inputs, update } = useCompressionState()

  const { result, computeError, invalidField } = useMemo((): {
    result: CompressionResult | null
    computeError: string | null
    invalidField: CompressionInputField | null
  } => {
    try {
      return { result: estimateCompression(toInputs(inputs)), computeError: null, invalidField: null }
    } catch (error) {
      return {
        result: null,
        computeError: error instanceof Error ? error.message : 'Those numbers cannot be costed.',
        invalidField: error instanceof CompressionInputError ? error.field : null,
      }
    }
  }, [inputs])

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="Context Compression Calculator" />

      <ToolHeader
        title="Context Compression Calculator"
        description="Find the point where summarising a session costs more than carrying it."
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <CompressionForm
          inputs={inputs}
          update={update}
          invalidField={invalidField}
          computeError={computeError}
        />
        <CompressionResults result={result} computeError={computeError} />
      </div>

      <FaqSection
        id="compression-faq-heading"
        heading="Questions about compressing a session"
        items={COMPRESSION_FAQ}
      />
    </div>
  )
}
