/**
 * Constants for the inference hardware model.
 *
 * Every number here is either a hardware fact or a stated allowance, so a
 * reader can check the model rather than take it on trust. Where a value is an
 * allowance rather than a measurement, the doc comment says so and the
 * calculator repeats it in its assumptions.
 */

import type { MtpHeadType } from './types'

/**
 * Bytes for each weight.
 *
 * FP16 and BF16 are both two bytes, so a model served in either one needs the
 * same VRAM and the same bandwidth. The two precisions differ in numeric range
 * and not in size, which is why this is a single constant and not a table.
 */
export const BYTES_PER_WEIGHT = 2

/** Bytes in a gibibyte, the unit the GPU presets are quoted in. */
export const GIB = 1024 ** 3

/**
 * The CUDA context, the framework, and the kernels that stay resident but are
 * not part of the model.
 */
export const RUNTIME_OVERHEAD_BYTES = 1.5 * GIB

/** Bytes for each element of a live activation. Two for both precisions. */
export const ACTIVATION_BYTES_PER_ELEMENT = 2

/**
 * How many live intermediates the activation buffer holds at once.
 *
 * A decode step keeps more than one tensor live: the query, key and value
 * projections, the attention output, and the two feed forward intermediates.
 * This is an estimate rather than a measurement, and the calculator says so.
 */
export const ACTIVATION_FACTOR = 3

/**
 * The share of a card's memory bandwidth a decode loop reaches.
 *
 * Decode reads every active weight once for each token, so it is bound by
 * bandwidth rather than by arithmetic. No real kernel reaches the quoted peak,
 * and 0.8 is a realistic ceiling for a well written attention kernel.
 */
export const BANDWIDTH_EFFICIENCY = 0.8

/**
 * The share of VRAM held back by default.
 *
 * A serving process cannot use the whole card. Fragmentation, the allocator,
 * and on a workstation card the display all take a cut, so a configuration that
 * fits exactly on paper does not fit in practice.
 */
export const DEFAULT_HEADROOM = 0.1

/** The most cards the ranking will use before it reports that nothing fits. */
export const MAX_SUGGESTED_GPUS = 8

/** The precision the page opens on. */
export const DEFAULT_PRECISION = 'BF16'

export const PRECISIONS = ['FP16', 'BF16'] as const

/**
 * One speculative decoding head, with the figure it is held to.
 *
 * `speedup` is the multiplier this calculator applies and `published` is the
 * figure the source reports. The applied value is deliberately the lower of
 * the two, because a published figure comes from one model on one benchmark.
 * `source` names where the published figure comes from, so a reader can check
 * it rather than take the number on trust.
 */
export interface MtpHeadSpec {
  id: MtpHeadType
  label: string
  /** Shown under the picker while this head is selected. */
  hint: string
  /** The multiplier the calculator applies. Always below the published figure. */
  speedup: number
  /** What the source actually measured. */
  published: string
  /** Where the published figure comes from. */
  source: string
}

/**
 * The heads on offer, in the order the picker shows them.
 *
 * Every applied multiplier is below the published figure, and the published
 * figure is stated beside it. A head is trained against one model, so this is
 * an estimate for measurement and not a promise about a given deployment.
 */
export const MTP_HEADS: readonly MtpHeadSpec[] = [
  {
    id: 'none',
    label: 'None',
    hint: 'No speculative head. The answer is the bandwidth roofline for the card.',
    speedup: 1,
    published: '1.0x, the baseline',
    source: 'the bandwidth roofline this calculator already reports',
  },
  {
    id: 'sequential-mtp',
    label: 'Sequential MTP',
    hint: 'The DeepSeek-V3 design. It chains one module for each extra token. A published 1.8x, so the answer uses 1.5x.',
    speedup: 1.5,
    published: '1.8x tokens each second at an 85 to 90 percent acceptance rate',
    source: 'DeepSeek-V3 technical report, arXiv 2412.19437',
  },
  {
    id: 'parallel-mtp',
    label: 'Parallel MTP heads',
    hint: 'Independent heads on one shared trunk. A published 3x at inference, so the answer uses 1.4x.',
    speedup: 1.4,
    published: 'up to 3x at inference with 4 token prediction',
    source: 'Better and Faster Large Language Models via Multi-token Prediction, arXiv 2404.19737',
  },
  {
    id: 'medusa',
    label: 'Medusa heads',
    hint: 'Extra decoding heads on a frozen backbone. A published 2.2x, so the answer uses 1.6x.',
    speedup: 1.6,
    published: 'over 2.2x with the backbone frozen',
    source: 'Medusa, arXiv 2401.10774',
  },
  {
    id: 'eagle-3',
    label: 'EAGLE-3 head',
    hint: 'A trained draft head at the feature level. A published 3.0x to 6.5x, so the answer uses 2.0x.',
    speedup: 2,
    published: '3.0x to 6.5x against vanilla decoding',
    source: 'EAGLE-3, arXiv 2503.01840',
  },
]

/** The head ids, for the URL parser and the validation guard. */
export const MTP_HEAD_IDS = MTP_HEADS.map((head) => head.id)

/** The head the page opens on, which is no head at all. */
export const DEFAULT_MTP_HEAD: MtpHeadType = 'none'

/** The row for a head id, or undefined when the id is unknown. */
export function mtpHeadSpec(id: MtpHeadType): MtpHeadSpec | undefined {
  return MTP_HEADS.find((head) => head.id === id)
}

/**
 * The multiplier a head applies to the decode rate.
 *
 * An unknown id falls back to no head rather than throwing, so a stale URL or
 * a stale caller cannot turn a missing row into a wrong answer.
 */
export function mtpSpeedup(id: MtpHeadType): number {
  return mtpHeadSpec(id)?.speedup ?? 1
}

/** True when a value is one of the head ids this calculator knows. */
export function isMtpHeadType(value: unknown): value is MtpHeadType {
  return typeof value === 'string' && (MTP_HEAD_IDS as readonly string[]).includes(value)
}
