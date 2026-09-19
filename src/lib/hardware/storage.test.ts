import { describe, expect, it } from 'vitest'

import { DEFAULT_STORAGE_ID, STORAGE_PRESETS, findStorage } from './storage'

describe('STORAGE_PRESETS', () => {
  it('offers at least three options with distinct bandwidths', () => {
    expect(STORAGE_PRESETS.length).toBeGreaterThanOrEqual(3)
    const bandwidths = STORAGE_PRESETS.map((storage) => storage.bandwidthGBs)
    expect(new Set(bandwidths).size).toBe(bandwidths.length)
  })

  it('gives every option a positive bandwidth', () => {
    for (const storage of STORAGE_PRESETS) {
      expect(storage.bandwidthGBs, storage.id).toBeGreaterThan(0)
    }
  })

  it('labels every option and explains it', () => {
    for (const storage of STORAGE_PRESETS) {
      expect(storage.label.trim(), storage.id).not.toBe('')
      expect(storage.note.trim(), storage.id).not.toBe('')
    }
  })

  it('covers host memory, NVMe, SATA, and a network share', () => {
    const ids = STORAGE_PRESETS.map((storage) => storage.id)
    expect(ids).toContain('pcie5-host-ram')
    expect(ids).toContain('nvme-pcie4')
    expect(ids).toContain('sata-ssd')
    expect(ids).toContain('network-10gbe')
  })

  it('ranks host memory above an NVMe drive above SATA', () => {
    const hostRam = findStorage('pcie5-host-ram')?.bandwidthGBs ?? 0
    const nvme = findStorage('nvme-pcie4')?.bandwidthGBs ?? 0
    const sata = findStorage('sata-ssd')?.bandwidthGBs ?? 0
    expect(hostRam).toBeGreaterThan(nvme)
    expect(nvme).toBeGreaterThan(sata)
  })
})

describe('findStorage', () => {
  it('returns the matching spec', () => {
    expect(findStorage('nvme-pcie4')?.bandwidthGBs).toBe(7)
  })

  it('returns undefined for an unknown id', () => {
    expect(findStorage('floppy')).toBeUndefined()
  })

  it('names a default that resolves', () => {
    expect(findStorage(DEFAULT_STORAGE_ID)).toBeDefined()
  })
})
