import { AlertTriangle, Clock, Cpu, HardDrive, Loader2, MemoryStick } from 'lucide-react'

import { formatBytes, formatDuration, formatExact } from '@/lib/format'
import type { GpuSpec } from '@/lib/hardware'
import type { DsparkResult, DsparkVerdict } from '@/lib/dspark'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'

import DsparkBreakdown from './DsparkBreakdown'
import type { DsparkShapeState } from './useDsparkShape'

const VERDICT_LABEL: Record<DsparkVerdict, string> = {
  fits: 'Fits',
  'needs-offline': 'Needs offline capture',
  'needs-more-gpus': 'Needs more cards',
  'needs-offload': 'Needs offloading',
}

function verdictClass(verdict: DsparkVerdict): string {
  if (verdict === 'fits') return 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
  if (verdict === 'needs-offline') return 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
  if (verdict === 'needs-more-gpus') return 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
  return 'bg-rose-500/15 text-rose-700 dark:text-rose-400'
}

interface DsparkResultsProps {
  result: DsparkResult | null
  shapeState: DsparkShapeState
  gpu: GpuSpec
  computeError: string | null
}

export default function DsparkResults({
  result,
  shapeState,
  gpu,
  computeError,
}: DsparkResultsProps) {
  if (shapeState.status === 'idle') {
    return (
      <Card>
        <CardContent className="text-muted-foreground py-10 text-center text-sm">
          Enter a target model to estimate the cost of training a drafter against it.
        </CardContent>
      </Card>
    )
  }

  if (shapeState.status === 'loading') {
    return (
      <Card>
        <CardContent
          className="text-muted-foreground flex items-center justify-center gap-2 py-10 text-sm"
          role="status"
        >
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          Reading the target config
        </CardContent>
      </Card>
    )
  }

  if (shapeState.status === 'error') {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>Could not read that target config</AlertTitle>
        <AlertDescription>
          <p>{shapeState.error}</p>
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
  const offline = result.mode === 'offline'
  // A hand described target has no family name to print, so the prose falls
  // back to a phrase rather than saying "against manual".
  const targetLabel =
    shape.modelType === 'manual' || shape.modelType === 'unknown'
      ? 'this target'
      : shape.modelType
  const boundLabel =
    result.bound === 'compute' ? 'the GPU doing the arithmetic' : 'storage feeding the cache back'
  const cachePerToken = formatBytes(result.cacheBytesPerToken)

  return (
    <div className="flex flex-col gap-4" aria-live="polite">
      {shape.looksLikeDraftConfig && (
        <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
          <AlertTriangle />
          <AlertTitle>This looks like a draft config, not a target</AlertTitle>
          <AlertDescription>
            <p>
              The config carries DSpark draft fields. Its{' '}
              {formatExact(shape.numLayers)} blocks are the draft's depth, and the
              target it was trained against is deeper, so every figure below that
              depends on the depth is understated. Cost the target's own config
              instead.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-muted-foreground text-xs tracking-wide uppercase">
                Estimated training run
              </p>
              <p
                className="mt-1 text-4xl font-medium tracking-tight tabular-nums"
                data-testid="dspark-duration"
              >
                {formatDuration(result.estimateSeconds)}
              </p>
            </div>
            <Badge className={verdictClass(result.verdict)}>
              {VERDICT_LABEL[result.verdict]} · {gpu.label}
            </Badge>
          </div>

          {/* A self contained answer, so it can be lifted on its own. */}
          <p className="text-sm">
            Training a DSpark drafter against {targetLabel} with{' '}
            {formatExact(shape.numLayers)} blocks and{' '}
            {formatExact(shape.hiddenSize)} hidden takes about{' '}
            <strong>{formatDuration(result.estimateSeconds)}</strong> on {gpu.label}
            {Number(result.vramBytes) > 0 && result.verdict !== 'fits'
              ? ` across ${result.gpusNeeded} card${result.gpusNeeded === 1 ? '' : 's'}`
              : ''}
            , and it is limited by {boundLabel}.
          </p>

          <dl className="border-border grid grid-cols-2 gap-x-4 gap-y-3 border-t pt-4 text-sm sm:grid-cols-3">
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Tokens per epoch</dt>
              <dd className="tabular-nums">{formatExact(result.trainingTokens)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Compute alone</dt>
              <dd className="tabular-nums">{formatDuration(result.computeSeconds)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Cache read alone</dt>
              <dd className="tabular-nums">{formatDuration(result.cacheReadSeconds)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Draft parameters</dt>
              <dd className="tabular-nums">{formatExact(result.draftParams)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Anchors per sequence</dt>
              <dd className="tabular-nums">
                {formatExact(result.numAnchors)}
                {result.anchorsClamped ? ' (capped)' : ''}
              </dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Target parameters</dt>
              <dd className="tabular-nums">{formatExact(shape.totalParams)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Overhead applied</dt>
              <dd className="tabular-nums">{result.overheadFactor}x</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <section className="flex flex-col gap-3" aria-labelledby="dspark-cache-heading">
        <h2 id="dspark-cache-heading" className="text-lg font-medium tracking-tight">
          How big is the target cache?
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
                  {offline
                    ? `${formatBytes(result.cacheBytes).text} of target hidden states`
                    : 'No cache, because the target is captured online'}
                </p>
                <p className="text-muted-foreground text-sm">
                  {offline ? (
                    <>
                      Each token stores {cachePerToken.text}: {formatExact(shape.hiddenSize)} values
                      from each of the {formatExact(result.numTargetLayers)} captured target layers
                      in bf16, the last hidden state that the distribution and confidence losses
                      both need, the token ids, and two masks. Over{' '}
                      {formatExact(result.trainingTokens)} tokens that is{' '}
                      {formatBytes(result.cacheBytes).text} written once, and read back once per
                      epoch.
                    </>
                  ) : (
                    <>
                      Online capture keeps the target resident and writes nothing, so the run costs
                      no storage at all. It costs {formatBytes(result.targetWeightBytes).text} of
                      VRAM for the target weights instead, which is what the memory verdict below
                      accounts for.
                    </>
                  )}
                </p>
                {offline && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    Capturing fewer target layers is the first lever if this does not fit your disk.
                    The cache is close to proportional to the captured layer count.
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="dspark-time-heading">
        <h2 id="dspark-time-heading" className="text-lg font-medium tracking-tight">
          How long does the run take?
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <Clock className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">
                  {formatDuration(result.computeSeconds)} of arithmetic
                  {offline
                    ? `, ${formatDuration(result.cacheReadSeconds)} of reading the cache back`
                    : ''}
                </p>
                <p className="text-muted-foreground text-sm">
                  Training scores {formatExact(result.positionsPerEpoch)} positions per epoch
                  against {formatExact(result.draftParams)} draft parameters, which is{' '}
                  {result.trainingFlops.toExponential(2)} FLOPs, plus{' '}
                  {result.contextFlops.toExponential(2)} for the context each block attends to
                  {offline ? ` and ${result.cachePrepFlops.toExponential(2)} to build the cache` : ''}.
                  The {result.overheadFactor}x overhead and setup land on top of the slower of the
                  two.
                </p>
                {result.anchorsClamped && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    The anchor count was capped at {formatExact(result.numAnchors)} per sequence,
                    one block per sequence token, because the requested count would score more
                    positions than the sequence holds. Raise the sequence length or lower the
                    anchor count to change this.
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="dspark-memory-heading">
        <h2 id="dspark-memory-heading" className="text-lg font-medium tracking-tight">
          Does it fit in VRAM?
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <MemoryStick
                className="text-muted-foreground mt-0.5 size-4 shrink-0"
                aria-hidden="true"
              />
              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">
                  {formatBytes(result.peakVramBytes).text} peak against{' '}
                  {formatBytes(result.vramBytes).text} on {gpu.label}
                </p>
                <p className="text-muted-foreground text-sm">
                  The drafter and its optimizer state are {formatBytes(result.draftWeightBytes).text}{' '}
                  and {formatBytes(result.optimizerBytes).text}, gradients are{' '}
                  {formatBytes(result.gradientBytes).text}, and the activation buffer is{' '}
                  {formatBytes(result.activationBytes).text}.
                  {offline
                    ? ' Offline capture means the target is not resident while the drafter trains, which is what makes one card enough.'
                    : ` Online capture keeps the ${formatBytes(result.targetWeightBytes).text} target resident for the whole run.`}
                </p>
                {result.verdict === 'needs-offline' && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    Precomputing the cache drops the target from memory and makes this fit. It costs
                    storage instead, which the cache section above sizes.
                  </p>
                )}
                {result.verdict === 'needs-more-gpus' && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    One card is not enough for the drafter and its state. Spread the run over{' '}
                    {result.gpusNeeded} cards, or lower the micro batch to shrink the activation
                    buffer.
                  </p>
                )}
                {result.verdict === 'needs-offload' && (
                  <p className="text-sm text-rose-700 dark:text-rose-400">
                    No reasonable number of cards holds this. Lower the micro batch, use fewer draft
                    layers, or offload the optimizer state to host memory.
                  </p>
                )}
                {result.verdict === 'fits' && (
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    This run fits. The cache still needs {formatBytes(result.cacheBytes).text} of
                    storage, and the target checkpoint needs its own space on top.
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="dspark-draft-heading">
        <h2 id="dspark-draft-heading" className="text-lg font-medium tracking-tight">
          What does the draft model cost?
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <Cpu className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">
                  {formatExact(result.draftParams)} parameters, about{' '}
                  {((result.draftParams / shape.totalParams) * 100).toFixed(1)} percent of the
                  target
                </p>
                <p className="text-muted-foreground text-sm">
                  {formatExact(result.draftBackboneParams)} in the backbone,{' '}
                  {formatExact(result.draftProjectionParams)} in the projection from the captured
                  target layers, {formatExact(result.draftMarkovParams)} in the Markov head, and{' '}
                  {formatExact(result.draftConfidenceParams)} in the confidence head. The embedding
                  and the language model head are shared with the target and frozen, so they are
                  never trained.
                </p>
                <p className="text-muted-foreground text-sm">
                  At inference the drafter proposes {formatExact(result.blockSize)} candidate tokens
                  per step, which the frozen target verifies in a single parallel pass. The
                  acceptance length you actually get depends on the data you trained on and on the
                  verification schedule you deploy, neither of which this calculator measures.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <DsparkBreakdown result={result} />
    </div>
  )
}
