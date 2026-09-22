import { describe, expect, it } from 'vitest'

import { DEFAULT_GPU_ID, GPU_PRESETS, GPU_VENDORS, findGpu } from './gpus'

/** Every card the catalog is expected to cover, by id. */
const EXPECTED_IDS = [
  // Datacenter.
  'b300',
  'b200',
  'h200',
  'h100',
  'rtx-pro-6000',
  // RTX 50 series.
  'rtx-5090',
  'rtx-5080',
  'rtx-5070-ti',
  'rtx-5070',
  'rtx-5060-ti',
  'rtx-5060',
  // RTX 40 series.
  'rtx-4090',
  'rtx-4080-super',
  'rtx-4080',
  'rtx-4070-ti-super',
  'rtx-4070-ti',
  'rtx-4070-super',
  'rtx-4070',
  'rtx-4060-ti',
  'rtx-4060',
  // RTX 30 series.
  'rtx-3090-ti',
  'rtx-3090',
  'rtx-3080-ti',
  'rtx-3080',
  'rtx-3070-ti',
  'rtx-3070',
  'rtx-3060-ti',
  'rtx-3060',
  // AMD datacenter.
  'mi300x',
  // RX 9000 series.
  'rx-9070-xt',
  'rx-9070',
  'rx-9060-xt',
  // RX 7000 series.
  'rx-7900-xtx',
  'rx-7900-xt',
  'rx-7800-xt',
  'rx-7700-xt',
  'rx-7600',
  // RX 6000 series.
  'rx-6800-xt',
  'rx-6700-xt',
  'rx-6600',
]

/** Architectures with no FP8 tensor path at all. */
const NO_FP8_IDS = [
  'rtx-3090-ti',
  'rtx-3090',
  'rtx-3080-ti',
  'rtx-3080',
  'rtx-3070-ti',
  'rtx-3070',
  'rtx-3060-ti',
  'rtx-3060',
  'rx-7900-xtx',
  'rx-7900-xt',
  'rx-7800-xt',
  'rx-7700-xt',
  'rx-7600',
  'rx-6800-xt',
  'rx-6700-xt',
  'rx-6600',
]

/** Architectures with a native FP4 dense tensor path. */
const FP4_GENERATIONS = ['Blackwell Ultra', 'Blackwell', 'RDNA 4']

/** Architectures that do have an FP8 tensor path. */
const FP8_IDS = [
  'b300',
  'b200',
  'h200',
  'h100',
  'rtx-pro-6000',
  'rtx-5090',
  'rtx-5080',
  'rtx-5070-ti',
  'rtx-5070',
  'rtx-5060-ti',
  'rtx-5060',
  'rtx-4090',
  'rtx-4080-super',
  'rtx-4080',
  'rtx-4070-ti-super',
  'rtx-4070-ti',
  'rtx-4070-super',
  'rtx-4070',
  'rtx-4060-ti',
  'rtx-4060',
  'mi300x',
  'rx-9070-xt',
  'rx-9070',
  'rx-9060-xt',
]

describe('GPU_PRESETS coverage', () => {
  it('includes every requested card', () => {
    const ids = new Set(GPU_PRESETS.map((gpu) => gpu.id))
    const missing = EXPECTED_IDS.filter((id) => !ids.has(id))
    expect(missing).toEqual([])
  })

  it('has no duplicate ids', () => {
    const ids = GPU_PRESETS.map((gpu) => gpu.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('marks FP8 support correctly across architectures', () => {
    for (const id of NO_FP8_IDS) {
      expect(findGpu(id)?.fp8DenseTflops, `${id} should have no FP8 path`).toBeNull()
    }
    for (const id of FP8_IDS) {
      const value = findGpu(id)?.fp8DenseTflops
      expect(value, `${id} should have an FP8 path`).not.toBeNull()
      expect(value ?? 0).toBeGreaterThan(0)
    }
  })

  it('marks FP4 support by architecture', () => {
    for (const gpu of GPU_PRESETS) {
      if (FP4_GENERATIONS.includes(gpu.generation)) {
        expect(gpu.fp4DenseTflops, `${gpu.id} should have an FP4 path`).not.toBeNull()
        expect(gpu.fp4DenseTflops ?? 0, gpu.id).toBeGreaterThan(0)
      } else {
        expect(gpu.fp4DenseTflops, `${gpu.id} should have no FP4 path`).toBeNull()
      }
    }
  })

  it('runs FP4 at twice the FP8 rate where both paths exist', () => {
    for (const gpu of GPU_PRESETS) {
      if (gpu.fp4DenseTflops === null) continue
      expect(gpu.fp8DenseTflops, gpu.id).not.toBeNull()
      expect(gpu.fp4DenseTflops, gpu.id).toBe((gpu.fp8DenseTflops ?? 0) * 2)
    }
  })

  it('never claims an FP4 path on a card with no FP8 path', () => {
    for (const gpu of GPU_PRESETS) {
      if (gpu.fp8DenseTflops === null) {
        expect(gpu.fp4DenseTflops, gpu.id).toBeNull()
      }
    }
  })

  it('covers both vendors', () => {
    const vendors = new Set(GPU_PRESETS.map((gpu) => gpu.vendor))
    for (const vendor of GPU_VENDORS) {
      expect(vendors.has(vendor)).toBe(true)
    }
  })
})

describe('GPU_PRESETS values', () => {
  it('gives every card a positive VRAM figure of at least 8 GiB', () => {
    for (const gpu of GPU_PRESETS) {
      expect(gpu.vramGiB, gpu.id).toBeGreaterThanOrEqual(8)
    }
  })

  it('gives every card positive bandwidth and BF16 throughput', () => {
    for (const gpu of GPU_PRESETS) {
      expect(gpu.bandwidthGBs, gpu.id).toBeGreaterThan(0)
      expect(gpu.bf16DenseTflops, gpu.id).toBeGreaterThan(0)
    }
  })

  it('keeps FP8 at or above the BF16 rate where it exists, since it is the narrower format', () => {
    for (const gpu of GPU_PRESETS) {
      if (gpu.fp8DenseTflops === null) continue
      expect(gpu.fp8DenseTflops, gpu.id).toBeGreaterThanOrEqual(gpu.bf16DenseTflops)
    }
  })

  it('labels every card and gives it a generation and a note', () => {
    for (const gpu of GPU_PRESETS) {
      expect(gpu.label.trim(), gpu.id).not.toBe('')
      expect(gpu.generation.trim(), gpu.id).not.toBe('')
      expect(gpu.note.trim(), gpu.id).not.toBe('')
    }
  })

  it('ranks the datacenter cards above the consumer cards on memory', () => {
    const b300 = findGpu('b300')
    const mi300x = findGpu('mi300x')
    const rtx5090 = findGpu('rtx-5090')
    expect(b300?.vramGiB).toBe(288)
    expect(mi300x?.vramGiB).toBe(192)
    expect(mi300x?.vramGiB).toBeGreaterThan(rtx5090?.vramGiB ?? 0)
    expect(b300?.vramGiB).toBeGreaterThan(mi300x?.vramGiB ?? 0)
  })
})

describe('findGpu', () => {
  it('returns the matching spec', () => {
    expect(findGpu('rtx-5090')?.label).toBe('RTX 5090')
    expect(findGpu('h200')?.vramGiB).toBe(141)
  })

  it('returns undefined for an unknown id', () => {
    expect(findGpu('rtx-9090')).toBeUndefined()
    expect(findGpu('')).toBeUndefined()
  })

  it('names a default that resolves', () => {
    expect(findGpu(DEFAULT_GPU_ID)).toBeDefined()
  })
})
