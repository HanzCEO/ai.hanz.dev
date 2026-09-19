import { AlertTriangle, Check, Copy, ExternalLink, Loader2 } from 'lucide-react'
import { useCallback, useState } from 'react'

import {
  PROVIDER_LIST,
  formatBytes,
  formatExact,
  type ComputeResult,
  type ModelConfigError,
  type Provider,
} from '@/lib/kvcache'
import { SUPPORT_URL } from '@/lib/seo'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import BreakdownPanel from '@/tools/kv-cache-calculator/BreakdownPanel'
import type { ModelConfigState } from '@/tools/kv-cache-calculator/useModelConfig'

function SupportButton() {
  return (
    <Button asChild variant="outline" size="sm">
      <a href={SUPPORT_URL} target="_blank" rel="noreferrer">
        Request support for this model
        <ExternalLink />
      </a>
    </Button>
  )
}

function ErrorState({
  error,
  onSwitchProvider,
}: {
  error: ModelConfigError
  onSwitchProvider: (provider: Provider) => void
}) {
  const other = error.provider === 'huggingface' ? 'modelscope' : 'huggingface'
  const otherLabel = PROVIDER_LIST.find((spec) => spec.id === other)?.label ?? other

  return (
    <Alert variant="destructive">
      <AlertTriangle />
      <AlertTitle>Could not read the model config</AlertTitle>
      <AlertDescription>
        <p>{error.message}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {error.suggestOtherProvider && (
            <Button variant="outline" size="sm" onClick={() => onSwitchProvider(other)}>
              Try {otherLabel}
            </Button>
          )}
          {error.kind === 'not_found' && (
            <Button asChild variant="outline" size="sm">
              <a href={SUPPORT_URL} target="_blank" rel="noreferrer">
                Ask for it to be added
                <ExternalLink />
              </a>
            </Button>
          )}
        </div>
      </AlertDescription>
    </Alert>
  )
}

interface CalculatorResultsProps {
  result: ComputeResult | null
  configState: ModelConfigState
  onSwitchProvider: (provider: Provider) => void
  computeError: string | null
}

export default function CalculatorResults({
  result,
  configState,
  onSwitchProvider,
  computeError,
}: CalculatorResultsProps) {
  const [copied, setCopied] = useState(false)

  const copy = useCallback(async () => {
    if (!result) return
    const size = formatBytes(result.totalBytes)
    const lines = [
      `${result.architecture.modelType}: ${size.text} (${size.exact} bytes)`,
      `context ${formatExact(result.contextLength)} x ${result.sequenceCount} sequence${result.sequenceCount === 1 ? '' : 's'}`,
      ...result.components.map(
        (component) => `${component.label}: ${formatExact(component.totalBytes)} bytes`,
      ),
    ]
    try {
      await navigator.clipboard.writeText(lines.join('\n'))
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // Clipboard access can be denied. The result is on screen either way.
    }
  }, [result])

  if (configState.status === 'idle') {
    return (
      <Card>
        <CardContent className="text-muted-foreground py-10 text-center text-sm">
          Enter a model id to size its KV cache.
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

  if (configState.status === 'error' && configState.error) {
    return <ErrorState error={configState.error} onSwitchProvider={onSwitchProvider} />
  }

  if (computeError) {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>Cannot size this model</AlertTitle>
        <AlertDescription>
          <p>{computeError}</p>
          <div className="mt-3">
            <SupportButton />
          </div>
        </AlertDescription>
      </Alert>
    )
  }

  if (!result) return null

  const size = formatBytes(result.totalBytes)
  const exceedsLimit =
    result.maxPositionEmbeddings !== null && result.contextLength > result.maxPositionEmbeddings

  return (
    <div className="flex flex-col gap-4" aria-live="polite">
      {result.bestEffort && (
        <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
          <AlertTriangle />
          <AlertTitle>Best effort estimate</AlertTitle>
          <AlertDescription>
            <p>
              This architecture is not one the calculator recognises. The shape was inferred from
              the config fields that were present, so check the number before sizing hardware on it.
            </p>
            <div className="mt-3">
              <SupportButton />
            </div>
          </AlertDescription>
        </Alert>
      )}

      {exceedsLimit && (
        <Alert className="border-amber-500/40 text-amber-700 dark:text-amber-400">
          <AlertTriangle />
          <AlertTitle>Beyond the model's context limit</AlertTitle>
          <AlertDescription>
            <p>
              This model lists a maximum of{' '}
              {result.maxPositionEmbeddings?.toLocaleString('en-US')} tokens, and you asked for{' '}
              {result.contextLength.toLocaleString('en-US')}. The size below assumes the model would
              accept the longer context.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardContent className="flex flex-col gap-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-muted-foreground text-xs tracking-wide uppercase">KV cache size</p>
              <p
                className="mt-1 text-4xl font-medium tracking-tight tabular-nums"
                data-testid="kv-cache-size"
              >
                {size.text}
              </p>
              <p
                className="text-muted-foreground mt-1 font-mono text-xs"
                data-testid="kv-cache-size-exact"
              >
                {size.exact} bytes
              </p>
            </div>
            <Button variant="ghost" size="sm" onClick={copy} aria-label="Copy the result">
              {copied ? <Check /> : <Copy />}
              {copied ? 'Copied' : 'Copy'}
            </Button>
          </div>

          <dl className="border-border grid grid-cols-2 gap-x-4 gap-y-2 border-t pt-4 text-sm sm:grid-cols-3">
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Per token</dt>
              <dd className="tabular-nums">{formatExact(result.bytesPerToken)} bytes</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Context</dt>
              <dd className="tabular-nums">{result.contextLength.toLocaleString('en-US')}</dd>
            </div>
            <div className="flex flex-col">
              <dt className="text-muted-foreground text-xs">Sequences</dt>
              <dd className="tabular-nums">{result.sequenceCount.toLocaleString('en-US')}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <BreakdownPanel result={result} />
    </div>
  )
}
