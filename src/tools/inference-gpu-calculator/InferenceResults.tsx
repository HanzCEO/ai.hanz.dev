import { AlertTriangle, Check, Copy, ExternalLink, Gauge, HardDrive, Loader2, Server, Sparkles } from 'lucide-react'
import { useCallback, useState } from 'react'

import { formatBytes, formatExact } from '@/lib/format'
import { mtpHeadSpec, type InferenceCandidate, type InferenceResult } from '@/lib/inference'
import type { ModelShape } from '@/lib/model-shape'
import { SUPPORT_URL } from '@/lib/seo'
import type { ConfigSourceState } from '@/lib/use-config-source'
import { weightFormatLabel } from '@/lib/weight-format'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

import InferenceBreakdown from './InferenceBreakdown'

const VERDICT_LABEL: Record<InferenceResult['verdict'], string> = {
  single: 'One card',
  multi: 'Several cards',
  none: 'Does not fit',
}

function verdictClass(verdict: InferenceResult['verdict']): string {
  if (verdict === 'single') return 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
  if (verdict === 'multi') return 'bg-sky-500/15 text-sky-700 dark:text-sky-400'
  return 'bg-rose-500/15 text-rose-700 dark:text-rose-400'
}

/** "1 x RTX 4090" or "4 x H100 SXM". */
function configurationLabel(candidate: InferenceCandidate): string {
  return `${candidate.gpuCount} x ${candidate.gpu.label}`
}

/** A number a reader can compare at a glance. */
function formatRate(tokensPerSecond: number): string {
  if (!Number.isFinite(tokensPerSecond) || tokensPerSecond <= 0) return 'no throughput'
  if (tokensPerSecond >= 100) return `${Math.round(tokensPerSecond).toLocaleString('en-US')} tokens/s`
  return `${tokensPerSecond.toFixed(1)} tokens/s`
}

interface InferenceResultsProps {
  result: InferenceResult | null
  configState: ConfigSourceState
  /** The shape read from the config, or null while there is none to read. */
  shape: ModelShape | null
  computeError: string | null
  /** The model as the reader named it. Falls back to the config model_type. */
  modelLabel?: string
}

export default function InferenceResults({
  result,
  configState,
  shape: detectedShape,
  computeError,
  modelLabel,
}: InferenceResultsProps) {
  const [copied, setCopied] = useState(false)

  const copy = useCallback(async () => {
    if (!result || !result.recommended) return
    const lines = [
      `${modelLabel ?? result.shape.modelType} in ${weightFormatLabel(result.weightFormat)}: ${configurationLabel(result.recommended)}`,
      `weights ${formatExact(result.weightsBytes)} bytes`,
      `kv cache ${formatExact(result.kvCacheBytes)} bytes`,
      `per card ${formatExact(result.recommended.perCardBytes)} bytes of ${formatExact(result.recommended.usableBytes)} usable`,
      `about ${Math.round(result.decodeTokensPerSecond)} tokens each second`,
    ]
    try {
      await navigator.clipboard.writeText(lines.join('\n'))
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // Clipboard access can be denied. The result is on screen either way.
    }
  }, [result, modelLabel])

  if (configState.status === 'idle') {
    return (
      <Card>
        <CardContent className="text-muted-foreground py-10 text-center text-sm">
          Enter a model to see the GPU configuration it needs.
        </CardContent>
      </Card>
    )
  }

  if (configState.status === 'loading') {
    return (
      <Card>
        <CardContent
          className="text-muted-foreground flex items-center justify-center gap-2 py-10 text-sm"
          role="status"
        >
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          Reading the model config
        </CardContent>
      </Card>
    )
  }

  if (configState.status === 'error') {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>The calculator cannot read that config</AlertTitle>
        <AlertDescription>
          <p>{configState.error?.message ?? 'The calculator cannot read that model config.'}</p>
        </AlertDescription>
      </Alert>
    )
  }

  if (!detectedShape) {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>The calculator cannot size that config</AlertTitle>
        <AlertDescription>
          <p>
            The config describes no transformer with a hidden size and a depth. The calculator
            therefore has nothing to size.
          </p>
        </AlertDescription>
      </Alert>
    )
  }

  if (computeError) {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>Those inputs cannot be costed</AlertTitle>
        <AlertDescription>
          <p>{computeError}</p>
        </AlertDescription>
      </Alert>
    )
  }

  if (!result) return null

  const shape = result.shape
  const name = modelLabel ?? shape.modelType
  const recommended = result.recommended
  const fitting: InferenceCandidate[] = recommended
    ? [recommended, ...result.alternatives]
    : []
  // The row behind the selected head, so the page can name it in prose rather
  // than repeat the label in every branch below.
  const mtp = result.mtpHead === 'none' ? null : (mtpHeadSpec(result.mtpHead) ?? null)
  const weightLabel = weightFormatLabel(result.weightFormat)
  const mixed = result.weightQuantization.mixed

  return (
    <div className="flex flex-col gap-4" aria-live="polite">
      {result.bestEffort && (
        <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
          <AlertTriangle />
          <AlertTitle>Best effort estimate</AlertTitle>
          <AlertDescription>
            <p>
              A value had to be inferred from the config rather than read from it. Check the
              figures before you size hardware on them.
            </p>
            {result.notes.length > 0 && (
              <ul className="mt-3 flex list-disc flex-col gap-1 pl-4 text-xs">
                {result.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            )}
          </AlertDescription>
        </Alert>
      )}

      {result.verdict === 'none' && (
        <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
          <AlertTriangle />
          <AlertTitle>No configuration in this list holds the model</AlertTitle>
          <AlertDescription>
            <p>
              The run needs {formatBytes(result.totalBytes).text} in total. No card up to{' '}
              {formatExact(result.maxGpus)} {result.maxGpus === 1 ? 'card' : 'cards'} holds it.
            </p>
            {result.closest && (
              <p className="mt-2">
                The nearest configuration is {configurationLabel(result.closest)}. One card would
                hold {formatBytes(result.closest.perCardBytes).text} against{' '}
                {formatBytes(result.closest.usableBytes).text} of usable VRAM.
              </p>
            )}
            <p className="mt-2">
              Three options remain. Quantize the weights to FP8 or a 4 bit format. Offload part of
              the model to host memory. Or raise the maximum GPU count above the limit set in the
              form.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-muted-foreground text-xs tracking-wide uppercase">
                Recommended configuration
              </p>
              <p
                className="mt-1 text-4xl font-medium tracking-tight tabular-nums"
                data-testid="inference-recommended"
              >
                {recommended ? configurationLabel(recommended) : 'Nothing fits'}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge className={verdictClass(result.verdict)}>{VERDICT_LABEL[result.verdict]}</Badge>
              {recommended && (
                <Button variant="ghost" size="sm" onClick={copy} aria-label="Copy the result">
                  {copied ? <Check /> : <Copy />}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              )}
            </div>
          </div>

          {/* A self contained answer, so it can be lifted on its own. */}
          <p className="text-sm">
            {recommended ? (
              <>
                <strong>{name}</strong> in <strong>{weightLabel}</strong> needs{' '}
                <strong>{configurationLabel(recommended)}</strong> for{' '}
                {formatExact(shape.totalParams)} parameters. The weights take{' '}
                {formatBytes(result.weightsBytes).text} and the cache takes{' '}
                {formatBytes(result.kvCacheBytes).text}. One card then holds{' '}
                {formatBytes(recommended.perCardBytes).text} of its{' '}
                {formatBytes(recommended.usableBytes).text} of usable VRAM, which reaches about{' '}
                {formatRate(result.decodeTokensPerSecond)}.
                {mtp && (
                  <>
                    {' '}
                    With {mtp.label} selected, the rate is {result.mtpSpeedup}x higher.
                  </>
                )}
              </>
            ) : (
              <>
                <strong>{name}</strong> in <strong>{weightLabel}</strong> does not fit any
                configuration in this list. The run needs{' '}
                {formatBytes(result.totalBytes).text} in total, which is{' '}
                {formatBytes(result.weightsBytes).text} of weights and{' '}
                {formatBytes(result.kvCacheBytes).text} of cache.
              </>
            )}
          </p>

          {/* A mixed checkpoint names two formats, so both are shown rather than one label. */}
          <dl className="border-border grid grid-cols-2 gap-x-4 gap-y-3 border-t pt-4 text-sm sm:grid-cols-3">
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Weight format</dt>
              <dd className="tabular-nums">{weightLabel}</dd>
            </div>
            {mixed && (
              <>
                <div className="flex flex-col">
                  <dt className="text-muted-foreground text-xs">Expert format</dt>
                  <dd className="tabular-nums">{weightFormatLabel(result.weightQuantization.experts)}</dd>
                </div>
                <div className="flex flex-col">
                  <dt className="text-muted-foreground text-xs">Dense format</dt>
                  <dd className="tabular-nums">{weightFormatLabel(result.weightQuantization.dense)}</dd>
                </div>
              </>
            )}
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Resident weights</dt>
              <dd className="tabular-nums">{formatBytes(result.weightsBytes).text}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">KV cache</dt>
              <dd className="tabular-nums">{formatBytes(result.kvCacheBytes).text}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Cache dtype</dt>
              <dd className="tabular-nums">{result.kvCacheDtype}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Cache for each token</dt>
              <dd className="tabular-nums">{formatExact(result.kvBytesPerToken)} bytes</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Activation buffer</dt>
              <dd className="tabular-nums">{formatBytes(result.activationBytes).text}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Framework reserve</dt>
              <dd className="tabular-nums">{formatBytes(result.runtimeReserveBytes).text}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Total needed</dt>
              <dd className="tabular-nums">{formatBytes(result.totalBytes).text}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      {recommended && (
        <section className="flex flex-col gap-3" aria-labelledby="inference-vram-heading">
          <h2 id="inference-vram-heading" className="text-lg font-medium tracking-tight">
            What each card holds
          </h2>
          <Card>
            <CardContent className="flex flex-col gap-4">
              <div className="flex items-start gap-3">
                <HardDrive
                  className="text-muted-foreground mt-0.5 size-4 shrink-0"
                  aria-hidden="true"
                />
                <div className="flex flex-col gap-1">
                  <p className="text-sm font-medium">
                    {configurationLabel(recommended)} holds{' '}
                    {formatBytes(recommended.perCardBytes).text} on each card
                  </p>
                  <p className="text-muted-foreground text-sm">
                    {recommended.gpuCount === 1
                      ? 'One card holds the whole model.'
                      : `The weights and the cache divide across the ${recommended.gpuCount} cards.`}{' '}
                    The activation buffer and the framework reserve stay in full on every card, so
                    a second card does not halve the footprint. This card offers{' '}
                    {formatBytes(recommended.usableBytes).text} once the headroom is held back.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      )}

      {recommended && (
        <section className="flex flex-col gap-3" aria-labelledby="inference-room-heading">
          <h2 id="inference-room-heading" className="text-lg font-medium tracking-tight">
            How much room is left?
          </h2>
          <Card>
            <CardContent className="flex flex-col gap-4">
              <div className="flex items-start gap-3">
                <Gauge className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
                <div className="flex flex-col gap-1">
                  <p className="text-sm font-medium">
                    {formatBytes(result.kvHeadroomBytes).text} free on one card
                  </p>
                  <p className="text-muted-foreground text-sm">
                    {result.maxContextAtSequences !== null &&
                    result.maxSequencesAtContext !== null ? (
                      <>
                        The cache and the activation buffer both grow with the context and the
                        sequence count. The cache divides across the tensor-parallel ranks, and the
                        buffer stays whole on every card. That room holds{' '}
                        {formatExact(result.maxContextAtSequences)} tokens of context at{' '}
                        {formatExact(result.sequences)}{' '}
                        {result.sequences === 1 ? 'sequence' : 'sequences'}. It also holds{' '}
                        {formatExact(result.maxSequencesAtContext)} sequences at{' '}
                        {formatExact(result.contextLength)} tokens each.
                      </>
                    ) : (
                      'The cache and the activation buffer both grow with the context and the sequence count. There is no room left for more.'
                    )}
                  </p>
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    A longer context or a higher sequence count raises the cache. The answer then
                    needs more cards or a larger card. Both are set in the form.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      )}

      {recommended && (
        <section className="flex flex-col gap-3" aria-labelledby="inference-rate-heading">
          <h2 id="inference-rate-heading" className="text-lg font-medium tracking-tight">
            How fast does it go?
          </h2>
          <Card>
            <CardContent className="flex flex-col gap-4">
              <div className="flex items-start gap-3">
                <Server className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
                <div className="flex flex-col gap-1">
                  <p className="text-sm font-medium">
                    About {formatRate(result.decodeTokensPerSecond)} while decoding
                  </p>
                  <p className="text-muted-foreground text-sm">
                    Each token reads {formatExact(shape.activeParamsPerToken)} active parameters
                    and the whole cache. Decode is therefore bound by memory bandwidth and not by
                    arithmetic. One sequence reaches about{' '}
                    {formatRate(result.perSequenceTokensPerSecond)}. The figure ignores prefill and
                    the cost of the interconnect.
                  </p>
                  {mtp ? (
                    <p className="text-muted-foreground text-sm">
                      That figure includes the {mtp.label}, which multiplies the roofline by{' '}
                      {result.mtpSpeedup}. Without a head the same configuration reaches about{' '}
                      {formatRate(result.baseDecodeTokensPerSecond)}. Each model needs its own
                      head, so this is an estimate for measurement.
                    </p>
                  ) : (
                    <p className="text-muted-foreground text-sm">
                      No MTP head is selected, so this is the bandwidth roofline for the card.
                      Each model needs its own head, so a head is a property of the checkpoint.
                    </p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      )}

      {fitting.length > 0 && (
        <section className="flex flex-col gap-3" aria-labelledby="inference-table-heading">
          <h2 id="inference-table-heading" className="text-lg font-medium tracking-tight">
            Every configuration that fits
          </h2>
          <Card className="gap-0 py-0">
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">
                  Each card model in the list, at the fewest cards that hold the run. The
                  recommended configuration is marked.
                </caption>
                <thead className="bg-muted/50 sticky top-0 text-left">
                  <tr>
                    <th scope="col" className="px-4 py-2 font-medium">
                      Configuration
                    </th>
                    <th scope="col" className="px-4 py-2 text-right font-medium">
                      VRAM each
                    </th>
                    <th scope="col" className="px-4 py-2 text-right font-medium">
                      Peak each
                    </th>
                    <th scope="col" className="px-4 py-2 text-right font-medium">
                      Decode
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {fitting.map((candidate, index) => (
                    <tr
                      key={`${candidate.gpu.id}-${candidate.gpuCount}`}
                      className="border-border border-t"
                    >
                      <td className="px-4 py-2">
                        <span className="flex items-center gap-2">
                          {configurationLabel(candidate)}
                          {index === 0 && <Badge variant="secondary">Recommended</Badge>}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {candidate.gpu.vramGiB} GB
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {formatBytes(candidate.perCardBytes).text}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {formatRate(candidate.decodeTokensPerSecond)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <p className="text-muted-foreground text-xs">
            Every card in the list appears at the fewest cards that hold the run, so a row with
            more cards is not a better answer. The list is ordered by card count, then by the size
            of one card, then by decode speed.
          </p>
        </section>
      )}

      <Card>
        <CardContent className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <Sparkles className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <p className="text-sm">
              If you want an end-to-end managed deployment of a model with full-cycle
              optimizations, contact us on Discord.
            </p>
          </div>
          <Button asChild size="sm" className="shrink-0">
            <a href={SUPPORT_URL} target="_blank" rel="noreferrer">
              Contact us on Discord
              <ExternalLink />
            </a>
          </Button>
        </CardContent>
      </Card>

      <Button asChild variant="outline" size="sm" className="w-fit">
        <a href={SUPPORT_URL} target="_blank" rel="noreferrer">
          Ask about a model or a card that is missing
          <ExternalLink />
        </a>
      </Button>

      <InferenceBreakdown result={result} />
    </div>
  )
}
