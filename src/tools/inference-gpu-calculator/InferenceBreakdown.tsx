import { formatBytes, formatExact } from '@/lib/format'
import { mtpHeadSpec, type InferenceResult } from '@/lib/inference'
import { weightFormatLabel } from '@/lib/weight-format'

import Breakdown, { BreakdownPanel, Row } from '@/components/breakdown/Breakdown'

/** A decode rate a reader can compare at a glance. */
function formatRate(tokensPerSecond: number): string {
  if (!Number.isFinite(tokensPerSecond) || tokensPerSecond <= 0) return 'no throughput'
  if (tokensPerSecond >= 100) return `${Math.round(tokensPerSecond).toLocaleString('en-US')} tokens/s`
  return `${tokensPerSecond.toFixed(1)} tokens/s`
}

export default function InferenceBreakdown({ result }: { result: InferenceResult }) {
  const recommended = result.recommended
  // Every head, including none, has a row, so this is never undefined.
  const mtp = mtpHeadSpec(result.mtpHead)
  // A selected head scales the decode rate, and the published figure it was held
  // to is the reason the reader picked it. That panel opens on arrival so the
  // provenance is on screen rather than one click away.
  const openPanels = result.mtpHead === 'none' ? [] : ['mtp']

  return (
    <Breakdown
      steps={result.steps}
      constants={result.constants}
      assumptions={result.assumptions}
      constantsLabel="Values this answer used"
      defaultOpenPanels={openPanels}
      extraPanels={
        <>
          <BreakdownPanel value="mtp" label="MTP head">
            <dl className="text-sm">
              <Row label="Head" value={mtp?.label ?? result.mtpHead} />
              <Row label="Multiplier" value={`${result.mtpSpeedup}x`} />
              <Row label="Roofline rate" value={formatRate(result.baseDecodeTokensPerSecond)} />
              <Row label="Rate with the head" value={formatRate(result.decodeTokensPerSecond)} />
              <Row label="Published figure" value={mtp?.published ?? 'not stated'} />
              <Row label="Source" value={mtp?.source ?? 'not stated'} />
            </dl>
            <p className="text-muted-foreground mt-3 text-xs">
              The roofline is the rate the card reaches with no speculative head. A head raises
              that rate, because one pass of the model then yields several accepted tokens. MTP
              is a property of the trained checkpoint. A head trained for one model does not
              transfer to another. The multiplier here is an estimate for measurement, and it is
              held below the published figure.
            </p>
          </BreakdownPanel>

          <BreakdownPanel value="memory" label="Memory in detail">
            <dl className="text-sm">
              <Row label="Weight format" value={weightFormatLabel(result.weightFormat)} />
              {result.weightQuantization.mixed && (
                <>
                  <Row
                    label="Expert format"
                    value={weightFormatLabel(result.weightQuantization.experts)}
                  />
                  <Row
                    label="Dense format"
                    value={weightFormatLabel(result.weightQuantization.dense)}
                  />
                </>
              )}
              <Row label="Resident weights" value={formatBytes(result.weightsBytes).text} />
              <Row
                label={`KV cache in ${result.kvCacheDtype}`}
                value={formatBytes(result.kvCacheBytes).text}
              />
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
