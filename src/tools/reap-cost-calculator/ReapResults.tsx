import { AlertTriangle, Clock, HardDrive, Loader2, Scissors } from 'lucide-react'

import type { GpuSpec } from '@/lib/hardware'
import { formatBytes, formatDuration, formatExact } from '@/lib/format'
import type { ReapResult, ReapVerdict } from '@/lib/reap'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'

import ReapBreakdown from './ReapBreakdown'
import type { ReapShapeState } from './useReapShape'

const VERDICT_LABEL: Record<ReapVerdict, string> = {
  'not-moe': 'Not applicable',
  fits: 'Fits',
  'needs-fp8': 'Needs FP8',
  'needs-offload': 'Needs offloading',
}

function verdictClass(verdict: ReapVerdict): string {
  if (verdict === 'fits') return 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
  if (verdict === 'needs-fp8') return 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
  if (verdict === 'needs-offload') return 'bg-rose-500/15 text-rose-700 dark:text-rose-400'
  return ''
}

function verdictSentence(result: ReapResult): string {
  const memory = formatBytes(result.vramBytes).text
  if (result.verdict === 'fits') {
    return `One expert block fits in ${memory}.`
  }
  if (result.verdict === 'needs-fp8') {
    return `One expert block fits only in FP8, at ${formatBytes(result.perMoELayerBytes / 2).text}.`
  }
  return `One expert block does not fit in ${memory}, even at the narrowest weight precision.`
}

interface ReapResultsProps {
  result: ReapResult | null
  shapeState: ReapShapeState
  gpu: GpuSpec
  computeError: string | null
}

export default function ReapResults({ result, shapeState, gpu, computeError }: ReapResultsProps) {
  if (shapeState.status === 'idle') {
    return (
      <Card>
        <CardContent className="text-muted-foreground py-10 text-center text-sm">
          Enter a mixture of experts model to estimate its REAP run.
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
          Reading the model config
        </CardContent>
      </Card>
    )
  }

  if (shapeState.status === 'error') {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>The calculator cannot read that config</AlertTitle>
        <AlertDescription>
          <p>{shapeState.error}</p>
        </AlertDescription>
      </Alert>
    )
  }

  if (shapeState.status === 'not-moe') {
    return (
      <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
        <AlertTriangle />
        <AlertTitle>REAP does not apply to this model</AlertTitle>
        <AlertDescription>
          <p>
            REAP removes routed experts from a sparsely activated mixture of experts. This model has
            no expert bank. Therefore there is nothing to prune, and there is no calibration to
            cost.
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

  if (!result || !result.moe || !result.shape) return null

  const shape = result.shape
  const boundLabel =
    result.bound === 'compute' ? 'the GPU that does the arithmetic' : 'the storage that feeds the weights'

  return (
    <div className="flex flex-col gap-4" aria-live="polite">
      <Card>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-muted-foreground text-xs tracking-wide uppercase">
                Estimated REAP run
              </p>
              <p
                className="mt-1 text-4xl font-medium tracking-tight tabular-nums"
                data-testid="reap-duration"
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
            REAP on {shape.modelType} takes about{' '}
            <strong>{formatDuration(result.estimateSeconds)}</strong> on a {gpu.label}. The model
            has {result.shape.numLayers.toLocaleString('en-US')} blocks, and each expert bank holds{' '}
            {shape.routedExperts.toLocaleString('en-US')} experts. The run follows your calibration
            recipe. {verdictSentence(result)} The run is limited by {boundLabel}.
          </p>

          <dl className="border-border grid grid-cols-2 gap-x-4 gap-y-3 border-t pt-4 text-sm sm:grid-cols-3">
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Calibration tokens</dt>
              <dd className="tabular-nums">{formatExact(result.tokens)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Arithmetic alone</dt>
              <dd className="tabular-nums">{formatDuration(result.computeSeconds)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Storage reads alone</dt>
              <dd className="tabular-nums">{formatDuration(result.streamSeconds)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Parameters per token</dt>
              <dd className="tabular-nums">{formatExact(shape.activeParamsPerToken)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Total parameters</dt>
              <dd className="tabular-nums">{formatExact(shape.totalParams)}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Overhead factor</dt>
              <dd className="tabular-nums">{result.overheadFactor}x</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <section className="flex flex-col gap-3" aria-labelledby="reap-prune-heading">
        <h2 id="reap-prune-heading" className="text-lg font-medium tracking-tight">
          Can I prune this model?
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
                  One expert block is {formatBytes(result.perMoELayerBytes).text}
                </p>
                <p className="text-muted-foreground text-sm">
                  Your GPU has {formatBytes(result.vramBytes).text} of VRAM. The observer holds 1
                  expert block plus a {formatBytes(result.activationBytes).text} activation buffer.
                  The peak is therefore {formatBytes(result.peakVramBytes).text}.
                </p>
                {result.verdict === 'needs-fp8' && (
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    Change the weight precision to FP8. The expert block then needs{' '}
                    {formatBytes(result.perMoELayerBytes / 2).text}, and the run becomes possible.
                  </p>
                )}
                {result.verdict === 'needs-offload' && (
                  <p className="text-sm text-rose-700 dark:text-rose-400">
                    No weight precision makes this expert block fit. Lower the micro batch, use a GPU
                    with more VRAM, or divide the calibration across several GPUs.
                  </p>
                )}
                {result.verdict === 'fits' && (
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    This run fits. The full set of weights still needs{' '}
                    {formatBytes(result.weightBytes).text} of storage or host memory.
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="reap-time-heading">
        <h2 id="reap-time-heading" className="text-lg font-medium tracking-tight">
          How long does a REAP run take?
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <Clock
                className="text-muted-foreground mt-0.5 size-4 shrink-0"
                aria-hidden="true"
              />
              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">
                  {formatDuration(result.computeSeconds)} of arithmetic and{' '}
                  {formatDuration(result.streamSeconds)} of storage reads
                </p>
                <p className="text-muted-foreground text-sm">
                  The calibration touches {formatExact(result.tokens)} tokens. Each token goes to{' '}
                  {result.shape.expertsPerToken} of {shape.routedExperts.toLocaleString('en-US')}{' '}
                  experts, so 1 token touches {formatExact(shape.activeParamsPerToken)} parameters.
                  The whole calibration needs {result.flops.toExponential(2)} FLOPs. The slower of the
                  2 values sets the duration. The {result.overheadFactor}x overhead and the setup
                  then apply.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="reap-size-heading">
        <h2 id="reap-size-heading" className="text-lg font-medium tracking-tight">
          How much smaller does it get?
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <Scissors
                className="text-muted-foreground mt-0.5 size-4 shrink-0"
                aria-hidden="true"
              />
              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">
                  {result.removedExperts === 0
                    ? `The run keeps all ${formatExact(shape.routedExperts)} experts. Nothing is removed at this setting.`
                    : `The run keeps ${formatExact(result.keptExperts)} of ${formatExact(shape.routedExperts)} experts. The model is ${result.reductionPercent.toFixed(1)} percent smaller.`}
                </p>
                <p className="text-muted-foreground text-sm">
                  {result.removedExperts === 0
                    ? `The expert parameters stay at ${formatBytes(shape.routedExpertParams * 2).text} in BF16, and the total parameter count is unchanged at ${formatExact(shape.totalParams)}.`
                    : `The expert parameters fall from ${formatBytes(shape.routedExpertParams * 2).text} to ${formatBytes(result.expertParamsAfter * 2).text} in BF16. The total parameters fall from ${formatExact(shape.totalParams)} to ${formatExact(result.totalParamsAfter)}.`}
                </p>
                <p className="text-sm text-amber-700 dark:text-amber-400">
                  {result.removedExperts === 0
                    ? 'Nothing is removed at this setting, so neither the VRAM nor the time changes.'
                    : result.activeParamsPerTokenAfter === shape.activeParamsPerToken
                      ? 'Active parameters per token are unchanged, because a token still goes to the same number of experts. REAP saves VRAM, but not time, unless the run also reduces the router top-k.'
                      : `Reducing the top-k to ${result.expertsPerTokenAfter} also cuts the active parameters to ${formatExact(result.activeParamsPerTokenAfter)}. That change is the source of the faster inference.`}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <ReapBreakdown result={result} />
    </div>
  )
}
