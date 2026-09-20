/**
 * Constants for the inference hardware model.
 *
 * Every number here is either a hardware fact or a stated allowance, so a
 * reader can check the model rather than take it on trust. Where a value is an
 * allowance rather than a measurement, the doc comment says so and the
 * calculator repeats it in its assumptions.
 */

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
