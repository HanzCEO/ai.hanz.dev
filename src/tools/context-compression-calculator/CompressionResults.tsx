import { AlertTriangle, Scissors } from 'lucide-react'

import { formatTokensShort, formatUsd } from '@/lib/format'
import type { CompressionResult } from '@/lib/compression'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'

import CompressionChart from './CompressionChart'
import CompressionBreakdown from './CompressionBreakdown'

interface CompressionResultsProps {
  result: CompressionResult | null
  computeError: string | null
}

/**
 * The answer, in one sentence: the session size at which a summary starts to
 * pay. When no session size inside the window reaches that point, the sentence
 * says so instead of naming a size the reader can never reach.
 */
function whenSentence(result: CompressionResult): string {
  const threshold = `${formatTokensShort(result.breakEvenSessionTokens)} tokens`
  const summaryCost = formatUsd(result.summaryCost)

  if (result.compressingWins) {
    return `Compress once the session passes ${threshold}, where carrying a ${result.compressionPercent} percent summary costs ${summaryCost}, the same as keeping the session.`
  }
  return `Compressing does not pay at a ${result.compressionPercent} percent cap, because the summary costs ${summaryCost} against ${formatUsd(result.keptAtWindow)} for carrying a full window.`
}

export default function CompressionResults({ result, computeError }: CompressionResultsProps) {
  if (computeError) {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>Those numbers do not describe a session</AlertTitle>
        <AlertDescription>
          <p>{computeError}</p>
        </AlertDescription>
      </Alert>
    )
  }

  if (!result) return null

  return (
    <div className="flex flex-col gap-6" aria-live="polite">
      <Card>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-muted-foreground text-xs tracking-wide uppercase">
                Compress once the session passes
              </p>
              <p
                className="mt-1 text-4xl font-medium tracking-tight tabular-nums"
                data-testid="compression-threshold"
              >
                {result.compressingWins
                  ? `${formatTokensShort(result.breakEvenSessionTokens)} tokens`
                  : 'never, at this cap'}
              </p>
            </div>
            <Badge
              className={
                result.compressingWins
                  ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
                  : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
              }
            >
              <Scissors aria-hidden="true" />
              {result.compressingWins ? 'Summarise and save' : 'Keep the whole session'}
            </Badge>
          </div>

          <p className="text-sm">{whenSentence(result)}</p>

          <p className="text-muted-foreground text-sm">
            {result.breakEvenPercent <= 100
              ? `A summary may keep up to ${result.breakEvenPercent.toFixed(1)} percent of the window and still pay off.`
              : 'Every summary cap up to a full window pays off here, because this session costs more per token than a summary does.'}
          </p>

          <dl className="border-border grid grid-cols-2 gap-x-4 gap-y-3 border-t pt-4 text-sm sm:grid-cols-3">
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Keep the session, full window</dt>
              <dd className="tabular-nums">{formatUsd(result.keptAtWindow)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">
                {`Carry a ${result.compressionPercent}% summary`}
              </dt>
              <dd className="tabular-nums">{formatUsd(result.summaryAtWindow)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Difference</dt>
              <dd className="tabular-nums">{formatUsd(Math.abs(result.savingAtWindow))}</dd>
            </div>
          </dl>

          <p className="text-muted-foreground text-xs">
            {`The summary costs the same at every session size, because it is capped at ${formatTokensShort(result.summaryTokens)} tokens.`}
          </p>
        </CardContent>
      </Card>

      <section className="flex flex-col gap-3" aria-labelledby="compression-chart-heading">
        <h2 id="compression-chart-heading" className="text-lg font-medium tracking-tight">
          Where the two costs cross
        </h2>
        <Card>
          <CardContent>
            <CompressionChart result={result} />
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="compression-detail-heading">
        <h2 id="compression-detail-heading" className="text-lg font-medium tracking-tight">
          What the rates are made of
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Kept rate, 1M session tokens</dt>
                <dd className="tabular-nums">{formatUsd(result.keptRatePerMillion)}</dd>
              </div>
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Summary rate, 1M summary tokens</dt>
                <dd className="tabular-nums">{formatUsd(result.summaryUnitPerMillion)}</dd>
              </div>
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Summary size</dt>
                <dd className="tabular-nums">{`${formatTokensShort(result.summaryTokens)} tokens`}</dd>
              </div>
            </dl>

            <div className="border-border border-t pt-4">
              <p className="text-muted-foreground text-xs">
                The shares your mix was read as, scaled so they add to 100 percent.
              </p>
              <dl className="mt-3 grid grid-cols-3 gap-x-4 text-sm">
                <div className="flex flex-col">
                  <dt className="text-muted-foreground text-xs">Miss</dt>
                  <dd className="tabular-nums">{`${result.normalized.missPercent.toFixed(2)}%`}</dd>
                </div>
                <div className="flex flex-col">
                  <dt className="text-muted-foreground text-xs">Cache</dt>
                  <dd className="tabular-nums">{`${result.normalized.cachePercent.toFixed(2)}%`}</dd>
                </div>
                <div className="flex flex-col">
                  <dt className="text-muted-foreground text-xs">Output</dt>
                  <dd className="tabular-nums">{`${result.normalized.outputPercent.toFixed(2)}%`}</dd>
                </div>
              </dl>
            </div>

            <CompressionBreakdown result={result} />
          </CardContent>
        </Card>
      </section>
    </div>
  )
}
