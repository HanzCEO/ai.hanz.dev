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
  /**
   * Blocks sampled per sequence per step. Carried by the preset because it has
   * to fit the sequence length: a recipe written for long sequences cannot be
   * pointed at short ones without lowering this.
   */
  numAnchors: number
  note: string
}

/** Tokens in one pass over the training set. */
export function presetTrainingTokens(preset: DsparkPreset): number {
  return preset.samples * preset.sequenceLength
}

/**
 * Training set sizes, from a smoke test to two published recipes.
 *
 * The last two are real runs whose settings their authors published, and they
 * differ in an instructive way. DeepSeek read 1.24 billion tokens ten times.
 * OpenBMB read 7.05 billion tokens six times, so 5.7 times as much unique data
 * with less repetition, and published the acceptance length it bought them.
 */
export const DSPARK_PRESETS: DsparkPreset[] = [
  {
    id: 'quick',
    label: 'Quick',
    samples: 25_000,
    sequenceLength: 2048,
    epochs: 3,
    numAnchors: 292,
    note: 'A smoke test, about 51 million tokens. It is enough to prove that the pipeline runs and that the drafter learns. It is not enough to produce a drafter worth shipping.',
  },
  {
    id: 'medium',
    label: 'Medium',
    samples: 125_000,
    sequenceLength: 4096,
    epochs: 5,
    numAnchors: 585,
    note: 'A middle setting, about 512 million tokens. That is roughly 0.4 of a pass over Open-PerfectBlend, which is enough for a usable drafter on one domain.',
  },
  {
    id: 'deepspec',
    label: 'DeepSpec paper recipe',
    samples: 302_000,
    sequenceLength: 4096,
    epochs: 10,
    numAnchors: 512,
    note: 'One pass over Open-PerfectBlend, about 1.24 billion tokens, read 10 times. The paper trained each drafter on this set. It writes roughly 38 TB of target cache against a Qwen3-4B target.',
  },
  {
    id: 'minicpm5-2b-dspark',
    label: 'MiniCPM5-2B-DSpark recipe',
    samples: 1_959_525,
    sequenceLength: 3600,
    epochs: 6,
    numAnchors: 514,
    note: 'The recipe OpenBMB published for openbmb/MiniCPM5-2B-DSpark: 1,959,525 sequences of about 3,600 tokens, which is 7.05 billion tokens for each pass, read 6 times. That is 42.3 billion tokens of arithmetic over 5.7 times the unique data of the paper recipe. Their published drafter checkpoint is 323,776,001 parameters and reaches an acceptance length of 5.52 at temperature 0.',
  },
]

export const DEFAULT_PRESET = 'minicpm5-2b-dspark'

export function findDsparkPreset(id: string): DsparkPreset | undefined {
  return DSPARK_PRESETS.find((preset) => preset.id === id)
}
