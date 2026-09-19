/**
 * Storage presets for the REAP cost calculator.
 *
 * Layer-wise calibration streams model weights from storage one block at a
 * time. When that stream is slower than the GPU can compute, the run becomes
 * I/O bound and the wall clock follows the bandwidth here rather than the
 * TFLOPS figure. This is why the same model takes far longer from an NVMe
 * drive than from memory, and why a network share can dominate everything.
 *
 * Bandwidths are decimal GB/s and describe the realistic ceiling of the link,
 * not the marketing peak of the device.
 */

export interface StorageSpec {
  id: string
  label: string
  /** Effective read bandwidth in GB/s, decimal gigabytes. */
  bandwidthGBs: number
  note: string
}

export const STORAGE_PRESETS: StorageSpec[] = [
  {
    id: 'pcie5-host-ram',
    label: 'Host RAM over PCIe 5.0 x16',
    bandwidthGBs: 55,
    note: 'Pinned host memory on a PCIe 5.0 slot. Fastest option that is not device memory.',
  },
  {
    id: 'pcie4-host-ram',
    label: 'Host RAM over PCIe 4.0 x16',
    bandwidthGBs: 25,
    note: 'Pinned host memory on a PCIe 4.0 slot.',
  },
  {
    id: 'nvme-pcie5',
    label: 'NVMe SSD over PCIe 5.0',
    bandwidthGBs: 14,
    note: 'A fast Gen5 drive, read sequentially. Sustained speed drops once the cache is exhausted.',
  },
  {
    id: 'nvme-pcie4',
    label: 'NVMe SSD over PCIe 4.0',
    bandwidthGBs: 7,
    note: 'A typical Gen4 drive, read sequentially.',
  },
  {
    id: 'sata-ssd',
    label: 'SATA SSD',
    bandwidthGBs: 0.55,
    note: 'SATA caps near 550 MB/s. Expect the run to be entirely I/O bound.',
  },
  {
    id: 'network-10gbe',
    label: 'Network share over 10 GbE',
    bandwidthGBs: 1.1,
    note: 'A 10 gigabit link, so about 1.1 GB/s at best. Shared with everything else on the link.',
  },
]

const STORAGE_BY_ID = new Map(STORAGE_PRESETS.map((storage) => [storage.id, storage]))

export function findStorage(id: string): StorageSpec | undefined {
  return STORAGE_BY_ID.get(id)
}

export const DEFAULT_STORAGE_ID = 'nvme-pcie4'
