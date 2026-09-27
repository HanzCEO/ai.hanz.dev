import { formatTokensShort, formatUsd } from '@/lib/format'
import type { CompressionResult } from '@/lib/compression'

import Breakdown, { BreakdownPanel, Row } from '@/components/breakdown/Breakdown'

/**
 * The working behind the crossing point, in the same accordion every other
 * calculator uses.
 *
 * The decision turns on two rates, one that climbs with the session and one
 * that does not, so the panel that matters is the one that splits each rate
 * into the parts it was built from and shows the size the summary is capped at.
 */
export default function CompressionBreakdown({ result }: { result: CompressionResult }) {
  return (
    <Breakdown
      steps={result.steps}
      constants={result.constants}
      assumptions={result.assumptions}
      constantsLabel="Values this used"
      extraPanels={
        <BreakdownPanel value="rates" label="The two rates in detail">
          <dl className="text-sm">
            <Row label="Miss part of the kept rate" value={formatUsd(result.keptParts.miss)} />
            <Row label="Cache part of the kept rate" value={formatUsd(result.keptParts.cache)} />
            <Row label="Output part of the kept rate" value={formatUsd(result.keptParts.output)} />
            <Row label="Kept rate" value={formatUsd(result.keptRatePerMillion)} />
            <Row label="Input part of the summary rate" value={formatUsd(result.summaryParts.input)} />
            <Row
              label="Output part of the summary rate"
              value={formatUsd(result.summaryParts.output)}
            />
            <Row label="Summary rate" value={formatUsd(result.summaryUnitPerMillion)} />
          </dl>
          <dl className="border-border mt-3 border-t pt-3 text-sm">
            <Row
              label="Summary size"
              value={`${formatTokensShort(result.summaryTokens)} tokens`}
            />
            <Row label="Cost of carrying it" value={formatUsd(result.summaryCost)} />
            <Row
              label="Session size where the two agree"
              value={`${formatTokensShort(result.breakEvenSessionTokens)} tokens`}
            />
          </dl>
          <p className="text-muted-foreground mt-3 text-xs">
            Each rate is dollars for 1M tokens of its own kind, so a session token
            and a summary token are not the same thing.
          </p>
        </BreakdownPanel>
      }
    />
  )
}
