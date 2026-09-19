import type { ArchitectureFamily, DtypeId } from '@/lib/kvcache'

export interface DefaultDtypes {
  kvCacheDtype: DtypeId
  indexerDtype: DtypeId
}

/**
 * The cache dtype a freshly selected model should open with.
 *
 * DeepSeek V4.1 was trained with an FP4 global cache: E2M1 main KV with one
 * E4M3 scale per 16 channels, and MXFP4 indexer keys with one UE8M0 scale per
 * 32 values. Opening on BF16 would report a cache roughly 3.6x larger than the
 * one the model actually ships, so the default follows the trained format.
 *
 * Every other family keeps BF16, which is what their published caches use.
 */
export function defaultDtypesForFamily(family: ArchitectureFamily): DefaultDtypes {
  if (family === 'dsv41') {
    return { kvCacheDtype: 'FP4', indexerDtype: 'FP4' }
  }
  return { kvCacheDtype: 'BF16', indexerDtype: 'BF16' }
}
