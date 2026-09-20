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
              ['For each token', formatBytes(result.cacheBytesPerToken).text],
              ['Captured layers', `${result.numTargetLayers}`],
              ['Whole target cache', offline ? formatBytes(result.cacheBytes).text : 'not written'],
              [
                'Write duration',
                offline ? `${(result.cacheWriteSeconds / 60).toFixed(1)} minutes` : 'nothing to write',
              ],
              [
                'Read duration',
                offline ? `${(result.cacheReadSeconds / 3600).toFixed(2)} hours` : 'nothing to read',
              ],
            ]}
            note={
              offline
                ? 'The run writes the target cache once during preparation. It reads the target cache back once in each epoch. The read duration therefore scales with the epoch count, and the write duration does not.'
                : 'Online capture never writes the target cache. None of these values apply. The target stays in VRAM instead.'
            }
          />
          <BreakdownItem
            value="draft"
            label="Drafter in detail"
            rows={[
              ['Backbone', formatBytes(result.draftBackboneParams * 2).text],
              ['Feature projection', formatBytes(result.draftProjectionParams * 2).text],
              ['Markov head', formatBytes(result.draftMarkovParams * 2).text],
              ['Confidence head', formatBytes(result.draftConfidenceParams * 2).text],
              ['Total in bf16', formatBytes(result.draftWeightBytes).text],
              ['Shared and frozen', formatBytes(result.shape.totalParams * 2).text],
            ]}
            note="The shared line is the target embedding and the language model head. The run loads them for the forward pass but never updates them. They are therefore not part of the trained parameter count."
          />
          <BreakdownItem
            value="memory"
            label="VRAM in detail"
            rows={[
              ['Drafter weights in bf16', formatBytes(result.draftWeightBytes).text],
              ['Optimizer state', formatBytes(result.optimizerBytes).text],
              ['Gradients', formatBytes(result.gradientBytes).text],
              [
                'Resident target',
                offline ? 'not in VRAM' : formatBytes(result.targetWeightBytes).text,
              ],
              ['Model state over the whole run', formatBytes(result.modelStateBytes).text],
              [
                result.gpuCount > 1
                  ? `Model state on each of the ${result.gpuCount} cards`
                  : 'Model state on the card',
                formatBytes(result.perCardStateBytes).text,
              ],
              ['Activation buffer for each card', formatBytes(result.activationBytes).text],
              ['Framework and kernels for each card', formatBytes(result.runtimeReserveBytes).text],
              ['Peak resident on each card', formatBytes(result.peakVramBytes).text],
              ['GPU VRAM', formatBytes(result.vramBytes).text],
            ]}
            note={
              offline
                ? 'The target ran once, while the run built the target cache. The target is not in VRAM during training, so the drafter and its state set the peak. The weights, the optimizer state and the gradients divide across the card count. The activation buffer and the framework reserve are held on every card and do not divide.'
                : 'Online capture keeps the target in VRAM for the whole run, so its weights are part of the model state. Offline capture removes that line. The model state divides across the card count, while the activation buffer and the framework reserve are held on every card.'
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
