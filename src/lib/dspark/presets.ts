/**
 * Defaults and recipes for the DSpark cost model.
 *
 * Every number here is traceable to a published source, named on the constant
 * that uses it, so a reader can check the model rather than take it on trust.
 * The two references are the DeepSpec training repository, which is DeepSeek's
 * own implementation of the paper, and the NeMo AutoModel DSpark recipe.
 */

/** Tokens drafted per block. gamma in the paper. */
export const DEFAULT_BLOCK_SIZE = 7

/** Draft backbone depth. The paper uses 5 for DSpark, against 1 for Eagle3. */
export const DEFAULT_DRAFT_LAYERS = 5

/** Blocks sampled from each sequence per training step. */
export const DEFAULT_NUM_ANCHORS = 512

/** Rank of the low-rank Markov head, W1 and W2 in the paper. */
export const DEFAULT_MARKOV_RANK = 256

/** Captured target layers. The paper takes five, spread across the depth. */
export const DEFAULT_TARGET_LAYERS = 5

/** Packed sequence length the target cache is built at. */
export const DEFAULT_SEQUENCE_LENGTH = 4096

/** Cards the reference configurations assume: a single node of eight. */
export const DEFAULT_GPU_COUNT = 8

/** Learning rate, from config/dspark/dspark_qwen3_4b.py. */
export const DEFAULT_LEARNING_RATE = 6.0e-4

/** Effective batch size across all workers, from the same config. */
export const DEFAULT_GLOBAL_BATCH_SIZE = 512

/** Warmup fraction of the schedule, from the same config. */
export const DEFAULT_WARMUP_RATIO = 0.04

/**
 * Dense tensor throughput is rarely reached. A third is a realistic target for
 * a small model with short blocks, where kernel launch overhead dominates.
 */
export const DEFAULT_MFU = 0.3

/**
 * Framework, data loader, and checkpointing overhead over the pure compute or
 * stream time. The FLOPs figure does not capture any of it.
 */
export const DEFAULT_OVERHEAD_FACTOR = 2

/** Loading the target, preparing the cache, and writing the draft checkpoint. */
export const DEFAULT_SETUP_SECONDS = 600

/** Sequences in flight at once. One keeps the activation buffer smallest. */
export const DEFAULT_MICRO_BATCH = 1

/** The framework, CUDA context, and kernels that are resident but uncounted. */
export const RUNTIME_OVERHEAD_BYTES = 1.5 * 1024 ** 3

/**
 * Activation bytes per element in the draft forward pass, over the product of
 * micro batch, anchors, block size and hidden width. Several intermediates are
 * live at once, so this is well above the two bytes of a single bf16 tensor.
 */
export const ACTIVATION_BYTES_PER_ELEMENT = 2

/** How many live intermediates the activation buffer holds at once. */
export const ACTIVATION_FACTOR = 6

/**
 * Bytes per parameter for the optimizer state.
 *
 * Two fp32 Adam moments, eight bytes. The bf16 weights and the gradients are
 * counted separately, so this is the optimizer's own footprint and nothing else.
 */
export const OPTIMIZER_BYTES_PER_PARAM = 8

/** Bytes per parameter for the gradients, in bf16. */
export const GRADIENT_BYTES_PER_PARAM = 2

/** Bytes per element for a bf16 weight or activation. */
export const BF16_BYTES = 2

/** Bytes per element for the int32 token ids in the cache. */
export const INT32_BYTES = 4

/** Bytes per element for the uint8 masks in the cache. */
export const UINT8_BYTES = 1

/**
 * The most cards the verdict will suggest before it stops recommending more
 * hardware and starts recommending offloading instead.
 */
export const MAX_SUGGESTED_GPUS = 8

export interface DsparkPreset {
  id: string
  label: string
  /** Sequences in the training set. */
  samples: number
  /** Packed length of each sequence. */
  sequenceLength: number
  /** Passes over the training set. */
  epochs: number
  note: string
}

/** Tokens in one pass over the training set. */
export function presetTrainingTokens(preset: DsparkPreset): number {
  return preset.samples * preset.sequenceLength
}

/**
 * Training set sizes, from a smoke test to the published recipe.
 *
 * The published setting is one pass over Open-PerfectBlend, which is about 1.24
 * billion tokens, read ten times. The DeepSpec README quotes roughly 38 TB for
 * the target cache at that setting with a Qwen3-4B target, which is the anchor
 * the engine is tested against.
 */
export const DSPARK_PRESETS: DsparkPreset[] = [
  {
    id: 'quick',
    label: 'Quick',
    samples: 25_000,
    sequenceLength: 2048,
    epochs: 3,
    note: 'A smoke test, about 51 million tokens. Enough to prove the pipeline runs and the draft is learning, not enough to produce a drafter worth shipping.',
  },
  {
    id: 'medium',
    label: 'Medium',
    samples: 125_000,
    sequenceLength: 4096,
    epochs: 5,
    note: 'A middle setting, about 512 million tokens. Roughly 0.4 of a pass over Open-PerfectBlend, enough for a usable drafter on one domain.',
  },
  {
    id: 'published',
    label: 'Published recipe',
    samples: 302_000,
    sequenceLength: 4096,
    epochs: 10,
    note: 'One pass over Open-PerfectBlend, about 1.24 billion tokens, read ten times, which is what the paper trained each drafter on. Roughly 38 TB of target cache against a Qwen3-4B target.',
  },
]

export const DEFAULT_PRESET = 'medium'

export function findDsparkPreset(id: string): DsparkPreset | undefined {
  return DSPARK_PRESETS.find((preset) => preset.id === id)
}
