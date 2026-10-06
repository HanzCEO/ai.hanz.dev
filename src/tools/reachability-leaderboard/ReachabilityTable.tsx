import { Fragment, useCallback, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

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

/**
 * The card tier a row runs on, in the one line the column has room for.
 *
 * The tier is named rather than a specific card, because several cards in the
 * directory share a memory size and the engine chose one of them by insertion
 * order. Naming a single card would state that it is the smallest one that
 * fits, which is false for every other card at the same size. The breakdown
 * names the concrete card the ranking recommended.
 */
function reachabilityLabel(row: ReachabilityRow): string {
  if (row.verdict === 'unreachable') return 'Not sized'
  if (row.verdict === 'none') return 'Needs more cards than the limit'
  if (row.verdict === 'single') {
    return row.smallestSingleGpu
      ? `${row.smallestSingleGpu.vramGiB} GB or larger`
      : 'Fits on one card'
  }
  return row.recommendedGpu
    ? `${row.gpuCount} cards, ${row.recommendedGpu.vramGiB} GB or larger`
    : `${row.gpuCount} cards`
}

/**
 * The badge tone for a verdict.
 *
 * A model this site could not size is not a hardware result, so it uses the
 * outline tone. Only a model that was costed and needs more cards than the
 * limit is a negative result about hardware.
 */
function verdictVariant(row: ReachabilityRow): 'secondary' | 'outline' | 'destructive' {
  if (row.verdict === 'unreachable') return 'outline'
  if (row.verdict === 'none') return 'destructive'
  if (row.verdict === 'multi') return 'outline'
  return 'secondary'
}

/** The card count cell, which is a plain dash when nothing was costed. */
function cardsCell(row: ReachabilityRow): string {
  return row.sizeable ? `${row.singleCardGpuCount} of ${row.consideredGpuCount}` : '-'
}

/** The decode cell, which is a dash when no configuration fits. */
function decodeCell(row: ReachabilityRow): string {
  if (!row.sizeable) return '-'
  return row.decodeTokensPerSecond > 0 ? formatExact(row.decodeTokensPerSecond) : '-'
}

/**
 * The leaderboard itself.
 *
 * One row per model, in the order the reader chose. A row opens onto its own
 * working, so the table stays readable and the detail stays collapsed. The first
 * paint renders the whole table from the baked dataset, with no network call,
 * which is what lets a crawler read it.
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
        Enter a context length, a sequence count, and a card limit to rank the models.
      </p>
    )
  }

  if (noGpuMatches) {
    return (
      <p className="text-muted-foreground text-sm">
        No card in the directory matches that class and vendor. Widen the filter to see the table.
      </p>
    )
  }

  if (rows.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        No model matches that search. {totalRows} models are listed.
      </p>
    )
  }

  const sizedCount = rows.filter((row) => row.sizeable).length

  return (
    <div className="flex flex-col gap-3">
      <p className="text-muted-foreground text-xs">
        {searching
          ? `${rows.length} of ${totalRows} models match.`
          : sizedCount === rows.length
            ? `${rows.length} models sized for this workload.`
            : `${sizedCount} of ${rows.length} models sized for this workload. The rest are listed with the reason.`}
      </p>

      {/* The height cap is for a phone, where the table would otherwise push the
          workload controls down the page. Paint containment is what stops the
          wide table inside from making the page itself scroll sideways: the
          scroll container clips it, but without containment the browser still
          reports the table as page overflow. */}
      <div className="max-h-[70vh] overflow-auto contain-paint lg:max-h-none">
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
              <th scope="col" className="py-2 pr-3 text-right font-medium whitespace-nowrap">
                Decision Index
              </th>
              <th scope="col" className="py-2 pr-3 font-medium whitespace-nowrap">
                Hardware needed
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium whitespace-nowrap">
                Single card
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium whitespace-nowrap">
                Tokens/s
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium whitespace-nowrap">
                Weights
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
                  <tr
                    data-engine={row.model.engine}
                    className="border-border border-b align-top"
                  >
                    <td className="text-muted-foreground py-3 pr-3 tabular-nums">{index + 1}</td>
                    <td className="py-3 pr-3">
                      <div className="font-medium">{row.model.name}</div>
                      <div className="text-muted-foreground text-xs">
                        {[row.model.engine, row.model.org, row.model.kind]
                          .filter(Boolean)
                          .join(' · ')}
                      </div>
                      {row.reasonShort && (
                        <div className="text-xs text-amber-600 dark:text-amber-500">
                          {row.reasonShort}
                        </div>
                      )}
                    </td>
                    <td className="text-muted-foreground py-3 pr-3 text-right tabular-nums whitespace-nowrap">
                      {row.model.index === null ? 'Not scored' : row.model.index.toFixed(2)}
                    </td>
                    <td className="py-3 pr-3">
                      <Badge variant={verdictVariant(row)}>{reachabilityLabel(row)}</Badge>
                    </td>
                    <td className="text-muted-foreground py-3 pr-3 text-right tabular-nums whitespace-nowrap">
                      {cardsCell(row)}
                    </td>
                    <td className="py-3 pr-3 text-right tabular-nums">{decodeCell(row)}</td>
                    <td className="text-muted-foreground py-3 pr-3 text-right tabular-nums whitespace-nowrap">
                      {row.sizeable ? formatBytes(row.weightsBytes).text : '-'}
                    </td>
                    <td className="py-3 text-right">
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 whitespace-nowrap"
                        aria-expanded={expanded}
                        aria-controls={
                          expanded ? `reachability-detail-${row.model.engine}` : undefined
                        }
                        onClick={() => toggle(row.model.engine)}
                      >
                        {expanded ? (
                          <ChevronDown aria-hidden="true" className="size-4" />
                        ) : (
                          <ChevronRight aria-hidden="true" className="size-4" />
                        )}
                        <span className="text-xs">{expanded ? 'Hide' : 'Details'}</span>
                      </button>
                    </td>
                  </tr>
                  {expanded && (
                    <tr
                      id={`reachability-detail-${row.model.engine}`}
                      className="border-border border-b"
                    >
                      <td colSpan={8} className="bg-muted/30 p-0">
                        <ReachabilityBreakdown row={row} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-muted-foreground text-xs">
        Tokens each second is the total across the concurrent sequences you set, for the
        configuration the ranking recommended.
      </p>
    </div>
  )
}
