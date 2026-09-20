import { formatBytes, formatExact } from '@/lib/format'
import type { ComputeResult } from '@/lib/kvcache'

import Breakdown, { Row } from '@/components/breakdown/Breakdown'

export default function BreakdownPanel({ result }: { result: ComputeResult }) {
  return (
    <Breakdown
      steps={result.steps}
      constants={result.constants}
      assumptions={result.assumptions}
      stepsExtra={
        <div className="mt-5 flex flex-col gap-3">
          {result.components.map((component) => (
            <div key={component.id} className="border-border rounded-lg border p-3">
              <div className="flex items-baseline justify-between gap-4">
                <span className="text-sm font-medium">{component.label}</span>
                <span className="text-sm tabular-nums">{formatBytes(component.totalBytes).text}</span>
              </div>
              <p className="text-muted-foreground mt-1 font-mono text-xs break-words">
                {component.formula}
              </p>
              <dl className="text-muted-foreground mt-2 text-xs">
                {component.bytesPerToken > 0 && (
                  <Row
                    label="per token, all layers"
                    value={`${formatExact(component.bytesPerToken)} bytes`}
                  />
                )}
                <Row
                  label="per sequence"
                  value={`${formatExact(component.bytesPerSequence)} bytes`}
                />
                <Row
                  label={`across ${result.sequenceCount} sequence${result.sequenceCount === 1 ? '' : 's'}`}
                  value={`${formatExact(component.totalBytes)} bytes`}
                />
              </dl>
              {component.note && (
                <p className="text-muted-foreground mt-2 text-xs">{component.note}</p>
              )}
            </div>
          ))}
        </div>
      }
    />
  )
}
