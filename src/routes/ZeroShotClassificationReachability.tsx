import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import { GPU_PRESETS } from '@/lib/hardware'
import {
  REACHABILITY_FAQ,
  REACHABILITY_MODELS,
  REACHABILITY_SOURCE,
} from '@/lib/reachability'
import ReachabilityForm from '@/tools/reachability-leaderboard/ReachabilityForm'
import ReachabilitySortPicker from '@/tools/reachability-leaderboard/ReachabilitySortPicker'
import ReachabilityTable from '@/tools/reachability-leaderboard/ReachabilityTable'
import { useReachabilityState } from '@/tools/reachability-leaderboard/useReachabilityState'

/** The date part of an ISO timestamp, or the value itself when it has none. */
function dateOnly(value: string): string {
  const at = value.indexOf('T')
  return at === -1 ? value : value.slice(0, at)
}

export default function ZeroShotClassificationReachability() {
  const { inputs, update, errors, valid, noGpuMatches, totalRows, rows } = useReachabilityState()

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="Zero-Shot Classification Reachability Leaderboard" />

      <ToolHeader
        title="Zero-Shot Classification Reachability Leaderboard"
        description={`Every model on the clef-evals Decision Model Leaderboard, ranked by the smallest card in the ${GPU_PRESETS.length} card hardware directory on this site that can hold it. The ranking is recalculated as you set the workload: the weights in the format each checkpoint publishes, the KV cache at your context length and sequence count, an activation buffer, and a framework reserve. Models this site cannot size stay in the table with the reason.`}
      />

      {/* The table comes first on a phone, so the result is not below a column of
          controls. The order is a view detail, so it is set on the container. */}
      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <div className="order-2 min-w-0 lg:order-1">
          <ReachabilityForm inputs={inputs} update={update} errors={errors} />
        </div>

        <section
          aria-labelledby="reachability-results-heading"
          className="order-1 flex min-w-0 flex-col gap-4 lg:order-2"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex flex-col gap-1">
              <h2 id="reachability-results-heading" className="text-lg font-medium tracking-tight">
                Results
              </h2>
              <p className="text-muted-foreground text-xs">
                Ordered by the hardware a model needs, so rank 1 needs the smallest card and not the
                highest Decision Index.
              </p>
            </div>
            <ReachabilitySortPicker value={inputs.sort} onChange={(sort) => update({ sort })} />
          </div>

          <ReachabilityTable
            rows={rows}
            totalRows={totalRows}
            valid={valid}
            noGpuMatches={noGpuMatches}
            searching={inputs.search.trim() !== ''}
          />
        </section>
      </div>

      <section
        aria-labelledby="reachability-source-heading"
        className="text-muted-foreground max-w-3xl text-sm"
      >
        <h2
          id="reachability-source-heading"
          className="text-foreground mb-2 text-lg font-medium tracking-tight"
        >
          Where the numbers come from
        </h2>
        <p>
          The model list, the Decision Index, and the measured latency come from the{' '}
          <a
            href={REACHABILITY_SOURCE.leaderboardUrl}
            className="text-primary underline-offset-4 hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            clef-evals Decision Model Leaderboard
          </a>
          , which is {REACHABILITY_SOURCE.upstreamLabel} and was generated on{' '}
          {dateOnly(REACHABILITY_SOURCE.upstreamGeneratedUtc)} on{' '}
          {REACHABILITY_SOURCE.upstreamHardware}. The checkpoint configs come from each model base
          checkpoint on HuggingFace, and the hardware figures come from the same {GPU_PRESETS.length}{' '}
          card directory the other calculators here use. This page was last built on{' '}
          {dateOnly(REACHABILITY_SOURCE.generatedUtc)}, from {REACHABILITY_MODELS.length} models.
        </p>
      </section>

      <FaqSection
        id="reachability-faq-heading"
        heading="Questions about reachability"
        items={REACHABILITY_FAQ}
      />
    </div>
  )
}
