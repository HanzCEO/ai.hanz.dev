import { Fragment, useCallback, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { formatBytes, formatExact } from '@/lib/format'
import type { ReachabilityRow } from '@/lib/reachability'

import ReachabilityBreakdown from './ReachabilityBreakdown'

interface ReachabilityTableProps {
  rows: ReachabilityRow[]
  /** Rows for the workload before the search box narrowed them. */
  totalRows: number
  /** False when a numeric field does not hold a usable value. */
  valid: boolean
  /** True when the card class and vendor filter select no card. */
  noGpuMatches: boolean
  /** True when a search is narrowing the table. */
  searching: boolean
}

/** The card a single card row runs on, or a shortfall, in one line. */
function reachabilityLabel(row: ReachabilityRow): string {
  if (row.verdict === 'unreachable') return 'Could not be sized'
  if (row.verdict === 'single') {
    return row.smallestSingleGpu
      ? `${row.smallestSingleGpu.label} or larger`
      : 'One card holds it'
  }
  if (row.verdict === 'multi') {
    return row.recommendedGpu
      ? `${row.gpuCount} x ${row.recommendedGpu.label}`
      : `${row.gpuCount} cards`
  }
  return 'Needs more cards than the limit'
}

function verdictVariant(row: ReachabilityRow): 'secondary' | 'outline' | 'destructive' {
  if (row.verdict === 'unreachable' || row.verdict === 'none') return 'destructive'
  if (row.verdict === 'multi') return 'outline'
  return 'secondary'
}

/**
 * The leaderboard itself.
 *
 * One row per model, in the order the reader chose. A row opens onto its own
 * working, so the table stays readable while the detail stays one click away.
 * The first paint renders the whole board from the baked dataset, with no
 * network call, which is what lets a crawler read the table.
 */
export default function ReachabilityTable({
  rows,
  totalRows,
  valid,
  noGpuMatches,
  searching,
}: ReachabilityTableProps) {
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set())

  const toggle = useCallback((engine: string) => {
    setOpen((current) => {
      const next = new Set(current)
      if (next.has(engine)) next.delete(engine)
      else next.add(engine)
      return next
    })
  }, [])

  if (!valid) {
    return (
      <p className="text-muted-foreground text-sm">
        Enter a context length, a sequence count, and a card limit to rank the board.
      </p>
    )
  }

  if (noGpuMatches) {
    return (
      <p className="text-muted-foreground text-sm">
        No card in the directory matches that class and vendor. Widen the filter to see the board.
      </p>
    )
  }

  if (rows.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        No model matches that search. {totalRows} models are on the board.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-muted-foreground text-xs">
        {searching
          ? `${rows.length} of ${totalRows} models match.`
          : `${rows.length} models ranked for this workload.`}
      </p>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">
            Zero-shot classification models ranked by the smallest card that holds each one
          </caption>
          <thead>
            <tr className="border-border text-muted-foreground border-b text-left text-xs">
              <th scope="col" className="py-2 pr-3 font-medium">
                #
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                Model
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Index
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                Runs on
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Cards
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Tokens/s
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Checkpoint
              </th>
              <th scope="col" className="py-2 font-medium">
                <span className="sr-only">Details</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const expanded = open.has(row.model.engine)
              return (
                <Fragment key={row.model.engine}>
                  <tr className="border-border border-b align-top">
                    <td className="text-muted-foreground py-3 pr-3 tabular-nums">{index + 1}</td>
                    <td className="py-3 pr-3">
                      <div className="font-medium">{row.model.name}</div>
                      <div className="text-muted-foreground text-xs">
                        {[row.model.engine, row.model.org, row.model.kind]
                          .filter(Boolean)
                          .join(' · ')}
                      </div>
                    </td>
                    <td className="py-3 pr-3 text-right tabular-nums">
                      {row.model.index === null ? 'Not scored' : row.model.index.toFixed(2)}
                    </td>
                    <td className="py-3 pr-3">
                      <Badge variant={verdictVariant(row)}>{reachabilityLabel(row)}</Badge>
                    </td>
                    <td className="text-muted-foreground py-3 pr-3 text-right tabular-nums">
                      {row.singleCardGpuCount} / {row.consideredGpuCount}
                    </td>
                    <td className="py-3 pr-3 text-right tabular-nums">
                      {row.sizeable && row.decodeTokensPerSecond > 0
                        ? formatExact(row.decodeTokensPerSecond)
                        : 'Not reachable'}
                    </td>
                    <td className="text-muted-foreground py-3 pr-3 text-right tabular-nums">
                      {row.sizeable ? formatBytes(row.weightsBytes).text : 'Not sized'}
                    </td>
                    <td className="py-3 text-right">
                      <button
                        type="button"
                        className="text-primary underline-offset-4 hover:underline"
                        aria-expanded={expanded}
                        aria-controls={`reachability-detail-${row.model.engine}`}
                        onClick={() => toggle(row.model.engine)}
                      >
                        {expanded ? 'Hide' : 'Details'}
                      </button>
                    </td>
                  </tr>
                  {expanded && (
                    <tr className="border-border border-b">
                      <td colSpan={8} className="bg-muted/30 p-0">
                        <div id={`reachability-detail-${row.model.engine}`}>
                          <ReachabilityBreakdown row={row} />
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
