import { formatBytes, formatExact } from '@/lib/format'
import type { InferenceResult } from '@/lib/inference'

import Breakdown, { BreakdownPanel, Row } from '@/components/breakdown/Breakdown'

export default function InferenceBreakdown({ result }: { result: InferenceResult }) {
  const recommended = result.recommended

  return (
    <Breakdown
      steps={result.steps}
      constants={result.constants}
      assumptions={result.assumptions}
      constantsLabel="Values this answer used"
      extraPanels={
        <>
          <BreakdownPanel value="memory" label="Memory in detail">
            <dl className="text-sm">
              <Row label="Resident weights" value={formatBytes(result.weightsBytes).text} />
              <Row label="KV cache" value={formatBytes(result.kvCacheBytes).text} />
              <Row label="Activation buffer" value={formatBytes(result.activationBytes).text} />
              <Row
                label="Framework reserve"
                value={formatBytes(result.runtimeReserveBytes).text}
              />
              <Row label="Total needed" value={formatBytes(result.totalBytes).text} />
            </dl>
            <p className="text-muted-foreground mt-3 text-xs">
              The weights and the cache divide across the tensor-parallel ranks. The activation
              buffer and the framework reserve stay in full on every card.
            </p>
          </BreakdownPanel>

          {recommended && (
            <BreakdownPanel value="card" label="What one card holds">
              <dl className="text-sm">
                <Row label="Configuration" value={`${recommended.gpuCount} x ${recommended.gpu.label}`} />
                <Row
                  label="Share of the weights"
                  value={formatBytes(result.weightsBytes / recommended.gpuCount).text}
                />
                <Row
                  label="Share of the cache"
                  value={formatBytes(result.kvCacheBytes / recommended.gpuCount).text}
                />
                <Row label="Activation buffer" value={formatBytes(result.activationBytes).text} />
                <Row
                  label="Framework reserve"
                  value={formatBytes(result.runtimeReserveBytes).text}
                />
                <Row label="Peak on one card" value={formatBytes(recommended.perCardBytes).text} />
                <Row label="Usable VRAM" value={formatBytes(recommended.usableBytes).text} />
                <Row label="Free after the run" value={formatBytes(recommended.headroomBytes).text} />
              </dl>
              <p className="text-muted-foreground mt-3 text-xs">
                The peak is {formatExact(recommended.perCardBytes)} bytes against{' '}
                {formatExact(recommended.usableBytes)} bytes of usable VRAM. The card itself holds{' '}
                {recommended.gpu.vramGiB} GB, and the rest is the headroom the form holds back.
              </p>
            </BreakdownPanel>
          )}
        </>
      }
    />
  )
}
