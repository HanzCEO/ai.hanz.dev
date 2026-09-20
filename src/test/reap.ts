import { findGpu, findStorage } from '@/lib/hardware'
import type { GpuSpec, StorageSpec } from '@/lib/hardware'
import {
  DEFAULT_MFU,
  DEFAULT_MICRO_BATCH,
  DEFAULT_OVERHEAD_FACTOR,
  DEFAULT_SETUP_SECONDS,
  type ReapInputs,
} from '@/lib/reap'

/**
 * A REAP input set shared by the engine tests and the results panel tests.
 *
 * Both suites need a valid, unremarkable configuration so that a test can
 * override exactly the one field it is about. Defining it once means a change
 * to a default cannot leave the two suites exercising different runs.
 */
export function reapInputs(overrides: Partial<ReapInputs> = {}): ReapInputs {
  return {
    calibrationSamples: 512,
    sequenceLength: 2048,
    pruneRatio: 0.4,
    gpu: findGpu('h200') as GpuSpec,
    storage: findStorage('nvme-pcie4') as StorageSpec,
    weightDtype: 'BF16',
    mfu: DEFAULT_MFU,
    overheadFactor: DEFAULT_OVERHEAD_FACTOR,
    setupSeconds: DEFAULT_SETUP_SECONDS,
    microBatchSize: DEFAULT_MICRO_BATCH,
    scaleTopK: false,
    ...overrides,
  }
}
