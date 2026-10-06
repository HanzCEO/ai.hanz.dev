import Breakdown, { Row } from '@/components/breakdown/Breakdown'
import { Badge } from '@/components/ui/badge'
import { formatBytes, formatExact } from '@/lib/format'
import type { InferenceCandidate } from '@/lib/inference'
import { REACHABILITY_SOURCE, type ReachabilityRow } from '@/lib/reachability'
import { weightFormatLabel } from '@/lib/weight-format'

/**
 * The configurations that fit, one line per card count.
 *
 * Every configuration with more cards than the recommendation also fits, so the
 * full list is a long tail of strictly worse options. The smallest card at each
 * card count carries the answer.
 */
function frontier(candidates: InferenceCandidate[]): InferenceCandidate[] {
  const smallestPerCount = new Map<number, InferenceCandidate>()
  for (const candidate of candidates) {
    const current = smallestPerCount.get(candidate.gpuCount)
    if (!current || candidate.gpu.vramGiB < current.gpu.vramGiB) {
      smallestPerCount.set(candidate.gpuCount, candidate)
    }
  }
  return [...smallestPerCount.values()].sort((a, b) => a.gpuCount - b.gpuCount)
}

/** One candidate, in the form the configurations list reads in. */
function configurationLine(candidate: InferenceCandidate): string {
  const cards = `${candidate.gpuCount} x ${candidate.gpu.label}`
  return `${cards}: ${formatBytes(candidate.perCardBytes).text} on each card against ${formatBytes(candidate.usableBytes).text} of usable VRAM`
}

/**
 * The working behind one row.
 *
 * A sized row gets the same three panels every calculator on this site uses,
 * because the engine already produced the steps, the constants, and the
 * assumptions. Above those sit the two things that are specific to this page:
 * where the model came from, and how the recommended cards compare with the
 * rest of the directory.
 *
 * A field the leaderboard does not carry is left out rather than printed as
 * "Not named", because most of the board does not carry it and a column of the
 * same word is noise.
 */
export default function ReachabilityBreakdown({ row }: { row: ReachabilityRow }) {
  const model = row.model
  const result = row.result

  const fits = result ? (result.recommended ? [result.recommended, ...result.alternatives] : []) : []
  const smallestPerCount = frontier(fits)

  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      <dl className="text-sm">
        <Row label="Method" value={model.kind} />
        <Row
          label="Base checkpoint"
          value={model.baseModel ?? (model.closed ? 'Closed hosted API' : 'Not published')}
        />
        {model.org && <Row label="Publisher" value={model.org} />}
        {model.source && model.source !== 'community' && (
          <Row label="Leaderboard source" value={model.source} />
        )}
        <Row
          label="Decision Index"
          value={model.index === null ? 'Not scored' : model.index.toFixed(2)}
        />
        {model.panelCoverage !== null && model.panelCoverage !== REACHABILITY_SOURCE.panelSize && (
          <Row label="Benchmarks scored" value={String(model.panelCoverage)} />
        )}
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
            <Badge variant={row.verdict === 'none' ? 'destructive' : 'secondary'}>
              {row.verdict === 'single'
                ? 'One card holds it'
                : row.verdict === 'multi'
                  ? `${row.gpuCount} cards hold it`
                  : 'No card within the limit holds it'}
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
              label="Decode rate, all sequences"
              value={`${formatExact(row.decodeTokensPerSecond)} tokens each second`}
            />
            <Row
              label="Decode rate, one sequence"
              value={`${formatExact(result.perSequenceTokensPerSecond)} tokens each second`}
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
              {result.recommended && <li>{configurationLine(result.recommended)}</li>}
              {smallestPerCount
                .filter((candidate) => candidate.gpuCount > row.gpuCount)
                .map((candidate) => (
                  <li key={candidate.gpuCount}>
                    {candidate.gpuCount} cards of {candidate.gpu.vramGiB} GB or larger
                  </li>
                ))}
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
            <p className="text-muted-foreground text-xs">
              {row.fittingGpuCount} of {row.consideredGpuCount} cards in the filter reach it, at up
              to {result.maxGpus} {result.maxGpus === 1 ? 'card' : 'cards'}.
            </p>
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
