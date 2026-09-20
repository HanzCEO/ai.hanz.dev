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
  'needs-more-gpus': 'Needs more GPUs',
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
          Enter a target to estimate the cost of training a drafter against it.
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
          Reading the config of the target
        </CardContent>
      </Card>
    )
  }

  if (shapeState.status === 'error') {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>The calculator cannot read the target config</AlertTitle>
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
    result.bound === 'compute'
      ? 'the GPU that does the arithmetic'
      : 'the storage that feeds the target cache back'
  const cachePerToken = formatBytes(result.cacheBytesPerToken)

  // The card count the panel reports. A run that fits is reported at the count
  // the user entered, so the sentence can never disagree with the verdict. A
  // run that does not fit is reported at the count it would take.
  const gpuSentence =
    result.verdict === 'needs-more-gpus'
      ? `The peak needs ${result.gpusNeeded} GPUs. `
      : result.verdict === 'needs-offload'
        ? result.gpusNeeded === null
          ? 'The activation buffer alone does not fit one card, so no card count holds this run. '
          : `The peak needs ${result.gpusNeeded} GPUs, which is more than this calculator will recommend. `
        : result.gpuCount > 1
          ? `The run divides across ${result.gpuCount} GPUs. `
          : ''

  return (
    <div className="flex flex-col gap-4" aria-live="polite">
      {shape.looksLikeDraftConfig && (
        <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
          <AlertTriangle />
          <AlertTitle>This config looks like a drafter checkpoint, not a target</AlertTitle>
          <AlertDescription>
            <p>
              The config carries DSpark drafter fields. Its{' '}
              {formatExact(shape.numLayers)} blocks are the depth of the drafter, and the target it
              was trained against is deeper. Every estimate below that depends on the depth is
              therefore too small. Cost the config of the target instead.
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
            Training a DSpark drafter against {targetLabel} takes about{' '}
            <strong>{formatDuration(result.estimateSeconds)}</strong> on {gpu.label}. The target has{' '}
            {formatExact(shape.numLayers)} blocks and {formatExact(shape.hiddenSize)} hidden.{' '}
            {gpuSentence}
            The run is limited by {boundLabel}.
          </p>

          <dl className="border-border grid grid-cols-2 gap-x-4 gap-y-3 border-t pt-4 text-sm sm:grid-cols-3">
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Tokens for each epoch</dt>
              <dd className="tabular-nums">{formatExact(result.trainingTokens)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Arithmetic alone</dt>
              <dd className="tabular-nums">{formatDuration(result.computeSeconds)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Target cache reads alone</dt>
              <dd className="tabular-nums">{formatDuration(result.cacheReadSeconds)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Drafter parameters</dt>
              <dd className="tabular-nums">{formatExact(result.draftParams)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Anchors for each sequence</dt>
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
              <dt className="text-muted-foreground text-xs">Overhead factor</dt>
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
                    : 'No target cache, because the run captures the target online'}
                </p>
                <p className="text-muted-foreground text-sm">
                  {offline ? (
                    <>
                      One token stores {cachePerToken.text}: {formatExact(shape.hiddenSize)} values
                      from each of the {formatExact(result.numTargetLayers)} captured target layers
                      in bf16. It also stores the last hidden state for the distribution and
                      confidence losses. The token ids and 2 masks come after that. Over{' '}
                      {formatExact(result.trainingTokens)} tokens, that is{' '}
                      {formatBytes(result.cacheBytes).text}. The run writes it once and reads it back
                      in each epoch.
                    </>
                  ) : (
                    <>
                      Online capture keeps the target in VRAM and writes nothing. The run therefore
                      needs no storage. It needs {formatBytes(result.targetWeightBytes).text} of
                      VRAM for the target weights instead. The VRAM verdict below accounts for that.
                    </>
                  )}
                </p>
                {offline && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    Capture fewer target layers if this does not fit your storage. The target cache
                    is close to proportional to the captured layer count.
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
                    ? ` and ${formatDuration(result.cacheReadSeconds)} of target cache reads`
                    : ''}
                </p>
                <p className="text-muted-foreground text-sm">
                  Training scores {formatExact(result.positionsPerEpoch)} positions in each epoch
                  against {formatExact(result.draftParams)} drafter parameters. That is{' '}
                  {result.trainingFlops.toExponential(2)} FLOPs. The context that each block attends
                  to adds {result.contextFlops.toExponential(2)}.
                  {offline
                    ? ` Building the target cache adds ${result.cachePrepFlops.toExponential(2)}.`
                    : ''}{' '}
                  The {result.overheadFactor}x overhead and the setup apply on top of the slower
                  value.
                </p>
                {result.anchorsClamped && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    The run capped the anchor count at {formatExact(result.numAnchors)} for each
                    sequence, which is 1 block for each sequence token. The sequence cannot hold the
                    requested count. Raise the sequence length or lower the anchor count to change
                    this.
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
                  {formatBytes(result.peakVramBytes).text} peak
                  {result.gpuCount > 1 ? ` for each of the ${result.gpuCount} cards` : ''} against{' '}
                  {formatBytes(result.vramBytes).text} of VRAM on {gpu.label}
                </p>
                <p className="text-muted-foreground text-sm">
                  The drafter and its optimizer state need{' '}
                  {formatBytes(result.draftWeightBytes).text} and{' '}
                  {formatBytes(result.optimizerBytes).text}. The gradients need{' '}
                  {formatBytes(result.gradientBytes).text}. The activation buffer needs{' '}
                  {formatBytes(result.activationBytes).text}.
                  {result.gpuCount > 1
                    ? ` The model state adds up to ${formatBytes(result.modelStateBytes).text}, and the run divides it so that each card holds ${formatBytes(result.perCardStateBytes).text}.`
                    : ''}
                  {offline
                    ? ` Offline capture keeps the target out of VRAM while the drafter trains, so the drafter and its state set the peak.${result.verdict === 'fits' && result.gpuCount === 1 ? ' That is why one GPU is enough.' : ''}`
                    : ` Online capture keeps the ${formatBytes(result.targetWeightBytes).text} target in VRAM for the whole run.`}
                </p>
                {result.verdict === 'needs-offline' && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    Writing the target cache first removes the target from VRAM and makes this fit. It
                    needs storage instead, which the target cache section above sizes.
                  </p>
                )}
                {result.verdict === 'needs-more-gpus' && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    {result.gpuCount === 1
                      ? 'One GPU is not enough for the drafter and its state.'
                      : `${result.gpuCount} GPUs are not enough for the drafter and its state.`}{' '}
                    Divide the run across {result.gpusNeeded} GPUs, or lower the micro batch to make
                    the activation buffer smaller.
                  </p>
                )}
                {result.verdict === 'needs-offload' && (
                  <p className="text-sm text-rose-700 dark:text-rose-400">
                    {result.gpusNeeded === null
                      ? 'The activation buffer and the framework reserve do not fit one card on their own, so no card count holds this run. Lower the micro batch first.'
                      : `The peak needs ${result.gpusNeeded} GPUs, which is more than this calculator will recommend. Lower the micro batch, use fewer draft layers, or move the optimizer state to host memory.`}
                  </p>
                )}
                {result.verdict === 'fits' && (
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    {offline
                      ? `This run fits. The target cache still needs ${formatBytes(result.cacheBytes).text} of storage, and the target checkpoint needs its own space on top.`
                      : 'This run fits. The run writes no target cache, so the target checkpoint is the only thing that needs storage.'}
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="dspark-draft-heading">
        <h2 id="dspark-draft-heading" className="text-lg font-medium tracking-tight">
          What does the drafter cost?
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
                  The backbone holds {formatExact(result.draftBackboneParams)}, and the projection
                  from the captured target layers holds {formatExact(result.draftProjectionParams)}.
                  The Markov head holds {formatExact(result.draftMarkovParams)}. The confidence head
                  holds {formatExact(result.draftConfidenceParams)}. The embedding and the language
                  model head are shared with the target and frozen. The run never trains them.
                </p>
                <p className="text-muted-foreground text-sm">
                  At inference time, the drafter proposes {formatExact(result.blockSize)} candidate
                  tokens at each step. The frozen target verifies them in 1 parallel pass. The
                  acceptance length depends on the training data and on the verification schedule you
                  deploy. This calculator measures neither one.
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
