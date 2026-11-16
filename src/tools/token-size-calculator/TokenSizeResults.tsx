import { AlertCircle } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import {
  bytesPerTokenForStorage,
  estimateTokenSize,
  formatTokenAmountShort,
  getStorageFormat,
  storageNote,
  tokenIdWidth,
  TOKENIZER_FAMILY_LABELS,
  type StorageFormatId,
  type TokenizerError,
  type TokenizerProfile,
} from '@/lib/token-size'
import type { TokenizerStatus } from '@/lib/use-tokenizer'

interface TokenSizeResultsProps {
  status: TokenizerStatus
  profile: TokenizerProfile | null
  url: string | null
  error: TokenizerError | null
  tokensValue: number | null
  storage: StorageFormatId
}

/** A URL shortened to what fits in a footnote. */
function shortUrl(url: string): string {
  return url.replace(/^https:\/\//, '').replace(/\/tokenizer\.json$/, '')
}

export default function TokenSizeResults({
  status,
  profile,
  url,
  error,
  tokensValue,
  storage,
}: TokenSizeResultsProps) {
  if (status === 'idle') {
    return (
      <p className="text-muted-foreground text-sm">
        Pick a model and enter a token count to size it.
      </p>
    )
  }

  if (status === 'loading') {
    return (
      <p className="text-muted-foreground text-sm">
        Reading the tokenizer for this model. A large vocabulary can take a moment.
      </p>
    )
  }

  if (status === 'error' || !profile) {
    return (
      <Alert variant="destructive">
        <AlertCircle />
        <AlertTitle>This tokenizer could not be read</AlertTitle>
        <AlertDescription>
          <p>{error?.message ?? 'The hub did not return a tokenizer for that model.'}</p>
        </AlertDescription>
      </Alert>
    )
  }

  if (tokensValue === null) {
    return (
      <Alert variant="destructive">
        <AlertCircle />
        <AlertTitle>The token count is not readable</AlertTitle>
        <AlertDescription>
          <p>Enter a whole number of tokens, or a number with a k, m, b, or t suffix.</p>
        </AlertDescription>
      </Alert>
    )
  }

  const format = getStorageFormat(storage)
  const bytesPerToken = bytesPerTokenForStorage(storage, profile)
  const result = estimateTokenSize(tokensValue, bytesPerToken)

  return (
    <div className="flex min-w-0 flex-col gap-6" aria-live="polite">
      <Card>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-muted-foreground text-xs tracking-wide uppercase">
                {formatTokenAmountShort(tokensValue)} tokens as {format.label}, using{' '}
                {TOKENIZER_FAMILY_LABELS[profile.family]}
              </p>
              <p
                className="mt-1 text-4xl font-medium tracking-tight tabular-nums"
                data-testid="token-size-headline"
              >
                {result.headline}
              </p>
            </div>
            <Badge variant="secondary" className="tabular-nums">
              {bytesPerToken.toFixed(3)} bytes per token
            </Badge>
          </div>

          <p className="text-sm">{storageNote(storage, profile)}</p>

          <p className="text-muted-foreground text-sm">
            Counted in binary that is <span className="tabular-nums">{result.binary}</span> (1 KiB =
            1,024 B), and the exact figure is{' '}
            <span className="tabular-nums">{result.exactBytes}</span> bytes.
          </p>
        </CardContent>
      </Card>

      <section aria-labelledby="token-size-units-heading" className="flex flex-col gap-3">
        <h2 id="token-size-units-heading" className="text-lg font-medium tracking-tight">
          The same size in every unit
        </h2>
        <Card>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
              {result.lines.map((line) => (
                <div key={line.unit} className="flex flex-col">
                  <dt className="text-muted-foreground text-xs">{line.unit}</dt>
                  <dd className="tabular-nums">{line.text}</dd>
                </div>
              ))}
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Bytes (exact)</dt>
                <dd className="tabular-nums">{result.exactBytes}</dd>
              </div>
            </dl>
            <p className="text-muted-foreground mt-4 text-xs">
              Decimal units: 1 KB = 1,000 B, 1 MB = 1,000 KB, 1 GB = 1,000 MB. The binary figure
              counts 1 KiB = 1,024 B, the way memory is quoted.
            </p>
          </CardContent>
        </Card>
      </section>

      <section
        aria-labelledby="token-size-details-heading"
        className="flex flex-col gap-3"
      >
        <h2 id="token-size-details-heading" className="text-lg font-medium tracking-tight">
          The tokenizer and format behind this figure
        </h2>
        <Card>
          <CardContent className="flex flex-col gap-4">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Format</dt>
                <dd>{format.label}</dd>
              </div>
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Family</dt>
                <dd>{TOKENIZER_FAMILY_LABELS[profile.family]}</dd>
              </div>
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Vocabulary</dt>
                <dd className="tabular-nums">{profile.vocabSize.toLocaleString('en-US')} tokens</dd>
              </div>
              <div className="flex flex-col">
                <dt className="text-muted-foreground text-xs">Bytes per token</dt>
                <dd className="tabular-nums">{bytesPerToken.toFixed(3)}</dd>
              </div>
              {storage !== 'text' && (
                <>
                  <div className="flex flex-col">
                    <dt className="text-muted-foreground text-xs">Text bytes per token</dt>
                    <dd className="tabular-nums">{profile.bytesPerToken.toFixed(3)}</dd>
                  </div>
                  <div className="flex flex-col">
                    <dt className="text-muted-foreground text-xs">Token id width</dt>
                    <dd className="tabular-nums">{tokenIdWidth(profile.vocabSize) * 8} bit</dd>
                  </div>
                </>
              )}
            </dl>

            {storage === 'text' && (
              <p className="text-muted-foreground text-sm">
                {profile.estimated
                  ? 'This family has no merge ranks to replay, so the figure is a vocabulary average rather than a measurement; treat the answer as a rough guide.'
                  : 'The figure is measured against a fixed sample corpus, so it is exact for that text and a fair basis for comparing this model with any other on this page.'}
              </p>
            )}
            {storage === 'parquet' && (
              <p className="text-muted-foreground text-sm">
                A Parquet file's size depends on how well its encoder compresses a column of
                adjacent token ids, so this figure is a rule of thumb. The Arrow figure is the
                uncompressed id width.
              </p>
            )}
          </CardContent>
        </Card>
      </section>

      {url && (
        <p className="text-muted-foreground text-xs">
          Read from{' '}
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="text-primary underline-offset-4 hover:underline"
          >
            {shortUrl(url)}
          </a>
        </p>
      )}
    </div>
  )
}