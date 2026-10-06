import Breakdown, { Row } from '@/components/breakdown/Breakdown'
import { Badge } from '@/components/ui/badge'
import { formatBytes, formatExact } from '@/lib/format'
import type { ReachabilityRow } from '@/lib/reachability'
import { weightFormatLabel } from '@/lib/weight-format'

/**
 * The working behind one row.
 *
 * A sized row gets the same three panels every calculator on this site uses,
 * because the engine already produced the steps, the constants, and the
 * assumptions. Above those sit the two things that are specific to this page:
 * where the model came from, and how the recommended cards compare with the
 * rest of the directory.
 */
export default function ReachabilityBreakdown({ row }: { row: ReachabilityRow }) {
  const model = row.model
  const result = row.result

  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      <dl className="text-sm">
        <Row label="Engine" value={model.engine} />
        <Row label="Publisher" value={model.org ?? 'Not named'} />
        <Row label="Method" value={model.kind} />
        <Row
          label="Base checkpoint"
          value={model.baseModel ?? (model.closed ? 'Closed hosted API' : 'Not published')}
        />
        <Row label="Leaderboard source" value={model.source ?? 'Not named'} />
        <Row
          label="Decision Index"
          value={model.index === null ? 'Not scored' : model.index.toFixed(2)}
        />
        <Row
          label="Benchmarks scored"
          value={model.panelCoverage === null ? 'Unknown' : String(model.panelCoverage)}
        />
      </dl>

      {model.modelUrl && (
        <p className="text-sm">
          <a
            href={model.modelUrl}
            className="text-primary underline-offset-4 hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            Open the model card
          </a>
        </p>
      )}

      {!row.sizeable || !result ? (
        <p className="text-muted-foreground text-sm">{row.reason}</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              {row.verdict === 'single'
                ? 'One card holds it'
                : row.verdict === 'multi'
                  ? `${row.gpuCount} cards hold it`
                  : 'No card holds it'}
            </Badge>
            <Badge variant="outline">
              {row.singleCardGpuCount} of {row.consideredGpuCount} cards hold it alone
            </Badge>
            <Badge variant="outline">
              {row.fittingGpuCount} of {row.consideredGpuCount} cards reach it at up to{' '}
              {result.maxGpus} cards
            </Badge>
          </div>

          <dl className="text-sm">
            <Row label="Weights" value={formatBytes(row.weightsBytes).text} />
            <Row label="KV cache" value={formatBytes(row.kvCacheBytes).text} />
            <Row label="Activation buffer" value={formatBytes(result.activationBytes).text} />
            <Row label="Framework reserve" value={formatBytes(result.runtimeReserveBytes).text} />
            <Row label="Total resident" value={formatBytes(row.totalBytes).text} />
            <Row
              label="Bytes each token reads"
              value={formatBytes(result.activeBytesPerToken + row.kvCacheBytes).text}
            />
            <Row
              label="Decode rate"
              value={`${formatExact(row.decodeTokensPerSecond)} tokens each second`}
            />
            <Row
              label="Weight format"
              value={row.weightFormat ? weightFormatLabel(row.weightFormat) : 'Unknown'}
            />
          </dl>

          <dl className="text-sm">
            <Row
              label="Parameters the config produced"
              value={row.configParams === null ? 'Unknown' : formatExact(row.configParams)}
            />
            <Row
              label="Parameters the leaderboard reports"
              value={row.publishedParams === null ? 'Not listed' : formatExact(row.publishedParams)}
            />
          </dl>
          <p className="text-muted-foreground text-xs">
            The size is config arithmetic, so the two counts can differ. A vision tower, a multi
            token prediction block, or a layer that shares weights is not counted separately.
          </p>

          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium">Configurations that fit</p>
            <ul className="text-muted-foreground flex flex-col gap-1 text-sm">
              {result.recommended && (
                <li>
                  {result.recommended.gpuCount} x {result.recommended.gpu.label}, the smallest
                  configuration, holding {formatBytes(result.recommended.perCardBytes).text} on
                  each card
                </li>
              )}
              {result.alternatives.slice(0, 8).map((candidate) => (
                <li key={candidate.gpu.id}>
                  {candidate.gpuCount} x {candidate.gpu.label}
                </li>
              ))}
              {result.alternatives.length > 8 && (
                <li>and {result.alternatives.length - 8} more configurations</li>
              )}
              {!result.recommended && (
                <li>
                  None. The closest configuration holds{' '}
                  {result.closest ? formatBytes(result.closest.perCardBytes).text : 'nothing'} on
                  each card, against{' '}
                  {result.closest ? formatBytes(result.closest.usableBytes).text : 'nothing'} of
                  usable VRAM.
                </li>
              )}
            </ul>
          </div>

          <Breakdown
            steps={result.steps}
            constants={result.constants}
            assumptions={result.assumptions}
            constantsLabel="Config values used"
          />
        </>
      )}
    </div>
  )
}
