import { describe, expect, it } from 'vitest'

import { loadConfigFixture } from '@/test/fixtures'

import { suggestedDtypeFor } from './useReapShape'

/**
 * The weight format the checkpoint itself declares.
 *
 * The calibration reads the whole checkpoint, but the VRAM peak is one expert
 * block, so the expert bucket is the format that decides the fit.
 */
describe('suggestedDtypeFor', () => {
  it('reads the MXFP4 experts of a MiMo-V2.6 checkpoint', () => {
    expect(suggestedDtypeFor(loadConfigFixture('mimo-v26-pro-rl'))).toBe('MXFP4')
    expect(suggestedDtypeFor(loadConfigFixture('mimo-v26-flash-rl'))).toBe('MXFP4')
  })

  it('reads the MXFP4 experts of a mixed DeepSeek V4 checkpoint', () => {
    expect(suggestedDtypeFor(loadConfigFixture('deepseek-v4-pro'))).toBe('MXFP4')
    expect(suggestedDtypeFor(loadConfigFixture('deepseek-v41-flash'))).toBe('MXFP4')
  })

  it('reads NVFP4 when the checkpoint names it', () => {
    const config = {
      model_type: 'test',
      quantization_config: { quant_method: 'nvfp4', store_dtype: 'nvfp4' },
    }
    expect(suggestedDtypeFor(config)).toBe('NVFP4')
  })

  it('keeps FP8 for an FP8 checkpoint', () => {
    const config = {
      model_type: 'test',
      quantization_config: { quant_method: 'fp8', fmt: 'e4m3', weight_block_size: [128, 128] },
    }
    expect(suggestedDtypeFor(config)).toBe('FP8_E4M3')
  })

  it('leaves a plain checkpoint to the default', () => {
    expect(suggestedDtypeFor(loadConfigFixture('qwen3-8b'))).toBeNull()
    expect(suggestedDtypeFor({})).toBeNull()
  })
})
