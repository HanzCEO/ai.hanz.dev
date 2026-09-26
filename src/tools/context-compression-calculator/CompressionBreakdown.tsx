import { formatUsd } from '@/lib/format'
import type { CompressionResult } from '@/lib/compression'

import Breakdown, { BreakdownPanel, Row } from '@/components/breakdown/Breakdown'

/**
 * The working behind the break-even share, in the same accordion every other
 * calculator uses.
 *
 * The two rates are linear in the context, so the only figures that decide the
 * answer are the two slopes. The panel that matters is therefore the one that
 * splits each slope into the parts it was built from.
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
            <Row label="Miss part of the kept rate" value={formatUsd(result.neverParts.miss)} />
            <Row label="Cache part of the kept rate" value={formatUsd(result.neverParts.cache)} />
            <Row label="Output part of the kept rate" value={formatUsd(result.neverParts.output)} />
            <Row label="Kept rate" value={formatUsd(result.neverRatePerMillion)} />
            <Row label="Input part of the summary rate" value={formatUsd(result.compressParts.input)} />
            <Row label="Output part of the summary rate" value={formatUsd(result.compressParts.output)} />
            <Row label="Summary rate" value={formatUsd(result.compressRatePerMillion)} />
          </dl>
          <p className="text-muted-foreground mt-3 text-xs">
            Each rate is dollars for 1M context tokens, so the two can be compared directly.
          </p>
        </BreakdownPanel>
      }
    />
  )
}
