import { describe, expect, it } from 'vitest'

import type { ArchitectureFamily } from '@/lib/kvcache'
import { defaultDtypesForFamily } from '@/tools/kv-cache-calculator/defaultDtypes'

const OTHER_FAMILIES: ArchitectureFamily[] = ['gqa', 'mla', 'dsv4', 'hybrid_linear', 'unknown']

describe('defaultDtypesForFamily', () => {
  it('opens DeepSeek V4.1 on the FP4 cache it was trained with', () => {
    expect(defaultDtypesForFamily('dsv41')).toEqual({
      kvCacheDtype: 'FP4',
      indexerDtype: 'FP4',
    })
  })

  it('keeps BF16 for every other family', () => {
    for (const family of OTHER_FAMILIES) {
      expect(defaultDtypesForFamily(family), family).toEqual({
        kvCacheDtype: 'BF16',
        indexerDtype: 'BF16',
      })
    }
  })

  it('only treats V4.1 as FP4, not V4', () => {
    expect(defaultDtypesForFamily('dsv4').kvCacheDtype).toBe('BF16')
    expect(defaultDtypesForFamily('dsv41').kvCacheDtype).toBe('FP4')
  })
})
