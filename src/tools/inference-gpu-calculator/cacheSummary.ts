import { formatBytes, formatExact } from '@/lib/format'
import type { ComputeResult } from '@/lib/kvcache'

/**
 * What step 1 decided, in one line.
 *
 * Once the cache step folds away, the reader still needs to see the model, the
 * context, the sequences, the cache dtype and the cache size it settled on. The
 * line is built here rather than in the page so a test can read it without
 * rendering.
 */
export function describeCacheStep(
  result: ComputeResult,
  modelId: string,
  kvCacheDtype: string,
): string {
  const sequences = `${formatExact(result.sequenceCount)} ${
    result.sequenceCount === 1 ? 'sequence' : 'sequences'
  }`
  return `${modelId} at ${formatExact(result.contextLength)} tokens and ${sequences} holds a cache of ${formatBytes(result.totalBytes).text} in ${kvCacheDtype}. Reopen step 1 to change the model, the context, the sequences, or the cache dtype.`
}
