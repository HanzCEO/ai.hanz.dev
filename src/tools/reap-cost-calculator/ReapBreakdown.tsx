import { formatBytes } from '@/lib/format'
import type { ReapResult } from '@/lib/reap'

import Breakdown, { BreakdownPanel, Row } from '@/components/breakdown/Breakdown'

export default function ReapBreakdown({ result }: { result: ReapResult }) {
  if (!result.moe || !result.shape) return null

  return (
    <Breakdown
      steps={result.steps}
      constants={result.constants}
      assumptions={result.assumptions}
      constantsLabel="Values read from the config"
      extraPanels={
        <BreakdownPanel value="memory" label="VRAM in detail">
          <dl className="text-sm">
            <Row label="One expert block" value={formatBytes(result.perMoELayerBytes).text} />
            <Row label="Activation buffer" value={formatBytes(result.activationBytes).text} />
            <Row
              label="Framework and kernels"
              value={formatBytes(result.peakVramBytes - result.perMoELayerBytes - result.activationBytes).text}
            />
            <Row label="Peak resident" value={formatBytes(result.peakVramBytes).text} />
            <Row label="GPU VRAM" value={formatBytes(result.vramBytes).text} />
            <Row label="All weights in storage" value={formatBytes(result.weightBytes).text} />
          </dl>
          <p className="text-muted-foreground mt-3 text-xs">
            The layer-wise observer holds 1 expert block at a time. The expert block and its
            activations therefore set the peak, and not the whole model. The full set of weights
            still needs storage.
          </p>
        </BreakdownPanel>
      }
    />
  )
}
