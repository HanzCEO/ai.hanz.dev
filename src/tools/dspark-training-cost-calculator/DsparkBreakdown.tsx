import { formatBytes } from '@/lib/format'
import type { DsparkResult } from '@/lib/dspark'

import Breakdown, { BreakdownPanel, Row } from '@/components/breakdown/Breakdown'

export default function DsparkBreakdown({ result }: { result: DsparkResult }) {
  const offline = result.mode === 'offline'

  return (
    <Breakdown
      steps={result.steps}
      constants={result.constants}
      assumptions={result.assumptions}
      constantsLabel="Values read from the config and the recipe"
      extraPanels={
        <>
          <BreakdownItem
            value="cache"
            label="Target cache in detail"
            rows={[
              ['Per token', formatBytes(result.cacheBytesPerToken).text],
              ['Captured layers', `${result.numTargetLayers}`],
              ['Whole cache', offline ? formatBytes(result.cacheBytes).text : 'not written'],
              ['Time to write', offline ? `${(result.cacheWriteSeconds / 60).toFixed(1)} minutes` : 'nothing to write'],
              ['Time to read back', offline ? `${(result.cacheReadSeconds / 3600).toFixed(2)} hours` : 'nothing to read'],
            ]}
            note={
              offline
                ? 'The cache is written once during preparation and read back once per epoch, which is why the read time scales with the epoch count and the write time does not.'
                : 'Online capture never materialises the cache, so none of these figures apply. The target stays resident instead.'
            }
          />
          <BreakdownItem
            value="draft"
            label="Draft model in detail"
            rows={[
              ['Backbone', formatBytes(result.draftBackboneParams * 2).text],
              ['Feature projection', formatBytes(result.draftProjectionParams * 2).text],
              ['Markov head', formatBytes(result.draftMarkovParams * 2).text],
              ['Confidence head', formatBytes(result.draftConfidenceParams * 2).text],
              ['Total, bf16', formatBytes(result.draftWeightBytes).text],
              ['Shared and frozen', formatBytes(result.shape.totalParams * 2).text],
            ]}
            note="The shared line is the target embedding and language model head. They are loaded for the forward pass but never updated, so they are not part of the trained parameter count."
          />
          <BreakdownItem
            value="memory"
            label="Memory in detail"
            rows={[
              ['Draft weights, bf16', formatBytes(result.draftWeightBytes).text],
              ['Optimizer state', formatBytes(result.optimizerBytes).text],
              ['Gradients', formatBytes(result.gradientBytes).text],
              ['Activation buffer', formatBytes(result.activationBytes).text],
              [
                'Resident target',
                offline ? 'not resident' : formatBytes(result.targetWeightBytes).text,
              ],
              ['Framework and kernels', formatBytes(result.runtimeReserveBytes).text],
              ['Peak resident', formatBytes(result.peakVramBytes).text],
              ['Card memory', formatBytes(result.vramBytes).text],
            ]}
            note={
              offline
                ? 'The target ran once, while the cache was being built, and is not resident during training. That is what keeps the peak low enough for a single card.'
                : 'Online capture keeps the target loaded for the whole run, so its weights are part of the peak. Switching to offline capture removes that line.'
            }
          />
        </>
      }
    />
  )
}

/** One extra panel, so the three sub-tables stay declarative. */
function BreakdownItem({
  value,
  label,
  rows,
  note,
}: {
  value: string
  label: string
  rows: Array<[string, string]>
  note: string
}) {
  return (
    <BreakdownPanel value={value} label={label}>
      <dl className="text-sm">
        {rows.map(([rowLabel, rowValue]) => (
          <Row key={rowLabel} label={rowLabel} value={rowValue} />
        ))}
      </dl>
      <p className="text-muted-foreground mt-3 text-xs">{note}</p>
    </BreakdownPanel>
  )
}
