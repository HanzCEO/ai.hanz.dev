import type { WeightDtype } from './types'

export interface WeightDtypeSpec {
  id: WeightDtype
  label: string
  bytes: number
  note: string
}

export const WEIGHT_DTYPES: WeightDtypeSpec[] = [
  { id: 'BF16', label: 'BF16', bytes: 2, note: 'Full precision weights. This precision gives the largest expert block in VRAM.' },
  { id: 'FP8', label: 'FP8', bytes: 1, note: 'Half the bytes of BF16. It needs a Blackwell, Hopper, CDNA 3, or RDNA 4 GPU.' },
  { id: 'INT4', label: 'INT4', bytes: 0.5, note: 'A quarter of the bytes of BF16. It needs a checkpoint that is already quantised.' },
]

export function bytesPerParam(dtype: WeightDtype): number {
  return WEIGHT_DTYPES.find((spec) => spec.id === dtype)?.bytes ?? 2
}

export interface CalibrationPreset {
  id: string
  label: string
  samples: number
  sequenceLength: number
  note: string
}

/**
 * The two published recipes plus a middle setting.
 *
 * The paper calibrated on 24576 samples at 16384 tokens, which is 402 million
 * tokens and by far the dominant cost. The vLLM recipe uses 512 samples at 2048
 * tokens, which is about 400 times cheaper and is enough to rank experts.
 */
export const CALIBRATION_PRESETS: CalibrationPreset[] = [
  {
    id: 'quick',
    label: 'Quick',
    samples: 512,
    sequenceLength: 2048,
    note: 'This recipe comes from the vLLM llm-compressor example. It uses about 1 million tokens.',
  },
  {
    id: 'medium',
    label: 'Medium',
    samples: 4096,
    sequenceLength: 8192,
    note: 'A middle setting. It uses about 34 million tokens. That is enough for a stable saliency ranking.',
  },
  {
    id: 'paper',
    label: 'Paper recipe',
    samples: 24576,
    sequenceLength: 16384,
    note: 'This is the calibration mix published with REAP. It uses about 403 million tokens.',
  },
]

export const DEFAULT_CALIBRATION_PRESET = 'quick'

export function findCalibrationPreset(id: string): CalibrationPreset | undefined {
  return CALIBRATION_PRESETS.find((preset) => preset.id === id)
}

/** Pruning ratios used in the paper and in published checkpoints. */
export const PRUNE_RATIOS = [0.25, 0.3, 0.4, 0.5]

export const DEFAULT_PRUNE_RATIO = 0.4

/** Dense tensor throughput is rarely reached. A third is a realistic target. */
export const DEFAULT_MFU = 0.3

/**
 * Framework, observer, and scheduler overhead over the pure compute or stream
 * time. Activation hooks, the saliency reduction, and the dataloader all add
 * work that the FLOPs figure does not capture.
 */
export const DEFAULT_OVERHEAD_FACTOR = 2

/** Loading the checkpoint, tokenising, and writing the pruned model. */
export const DEFAULT_SETUP_SECONDS = 600

/** Samples in flight at once. One keeps the activation buffer smallest. */
export const DEFAULT_MICRO_BATCH = 1

/** The framework, CUDA context, and kernels that are resident but uncounted. */
export const RUNTIME_OVERHEAD_BYTES = 1.5 * 1024 ** 3

/** Weight formats from widest to narrowest, for the fitting search. */
export const WEIGHT_DTYPE_ORDER: WeightDtype[] = ['BF16', 'FP8', 'INT4']
