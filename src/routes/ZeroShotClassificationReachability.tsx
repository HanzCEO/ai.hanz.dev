import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import { GPU_PRESETS } from '@/lib/hardware'
import {
  REACHABILITY_FAQ,
  REACHABILITY_SOURCE,
  DEFAULT_REACHABILITY_HEADROOM,
} from '@/lib/reachability'
import ReachabilityForm from '@/tools/reachability-leaderboard/ReachabilityForm'
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
        description={`Every model on the clef-evals Decision Model Leaderboard, ranked by the smallest card in the ${GPU_PRESETS.length} card hardware directory on this site that can hold it. Each model is costed at a workload you choose: the checkpoint weights in the format it publishes, the KV cache at your context length and sequence count, an activation buffer, and a framework reserve of ${DEFAULT_REACHABILITY_HEADROOM * 100} percent of the card. The table reports the smallest card that holds the run, how many cards hold it on their own, and the decode rate that configuration reaches. Models the site cannot size stay on the board with the reason, so the table never overstates how much of the board the hardware reaches.`}
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <ReachabilityForm inputs={inputs} update={update} errors={errors} />

        <section aria-labelledby="reachability-board-heading" className="flex flex-col gap-4">
          <h2 id="reachability-board-heading" className="text-lg font-medium tracking-tight">
            The board
          </h2>
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
        <h2 id="reachability-source-heading" className="text-foreground mb-2 text-lg font-medium tracking-tight">
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
          checkpoint on HuggingFace. This page was last baked on{' '}
          {dateOnly(REACHABILITY_SOURCE.generatedUtc)}.
        </p>
        <p className="mt-2">
          The hardware figures come from the same GPU directory the other calculators on this site
          use, which is {GPU_PRESETS.length} cards from 8 GB to 288 GB.
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
