import { formatBytes, formatExact } from '@/lib/format'
import type { ComputeResult } from '@/lib/kvcache'

/**
 * What step 1 decided, in one line.
 *
 * Once the cache step folds away, the reader still needs to see the model, the
 * context, the sequences, the cache dtypes, and the cache size it settled on.
 * The line is built here rather than in the page so a test can read it without
 * rendering.
 *
 * A model with a sparse indexer holds two caches, and the two can be set to
 * different dtypes. Naming only one would misreport a mixed pair, so the two
 * are named whenever they differ. The indexer is left out for a model that has
 * none, because the field does not exist for it.
 */
export function describeCacheStep(
  result: ComputeResult,
  modelId: string,
  kvCacheDtype: string,
  indexerDtype: string = kvCacheDtype,
): string {
  const sequences = `${formatExact(result.sequenceCount)} ${
    result.sequenceCount === 1 ? 'sequence' : 'sequences'
  }`
  const head = `${modelId} at ${formatExact(result.contextLength)} tokens and ${sequences} holds a cache of ${formatBytes(result.totalBytes).text}`

  const mixed = result.indexerDtypeSupport !== null && indexerDtype !== kvCacheDtype
  const sizes = mixed
    ? `${head}. The attention cache is in ${kvCacheDtype} and the indexer cache is in ${indexerDtype}.`
    : `${head} in ${kvCacheDtype}.`

  return `${sizes} Reopen step 1 to change the model, the context, the sequences, or the cache dtypes.`
}
