import type { WeightFormatId, WeightFormatSpec } from './types'

/**
 * Every weight format the calculators can price.
 *
 * Each byte figure counts the payload and the scale sidecar, because a 4 bit
 * checkpoint is not half the size of an 8 bit one. MXFP4 shares one E8M0 scale
 * across 32 values, so one value costs 4.25 bits. NVFP4 shares one E4M3 scale
 * across 16 values, so one value costs 4.5 bits. The gaps look small against
 * the payload and they are not: they move a trillion parameter checkpoint by
 * tens of gigabytes.
 */
export const WEIGHT_FORMATS: WeightFormatSpec[] = [
  {
    id: 'BF16',
    label: 'BF16',
    bytes: 2,
    blockSize: null,
    note: 'Two bytes for each weight and no scale. The published format of most checkpoints.',
  },
  {
    id: 'FP16',
    label: 'FP16',
    bytes: 2,
    blockSize: null,
    note: 'Two bytes for each weight and no scale. The same size as BF16.',
  },
  {
    id: 'FP8_E4M3',
    label: 'FP8 (E4M3)',
    bytes: 1 + 1 / 16384,
    blockSize: 16384,
    note: 'One byte for each weight. A block scaled checkpoint adds one UE8M0 scale for each block, which is a fraction of a percent.',
  },
  {
    id: 'FP8_E5M2',
    label: 'FP8 (E5M2)',
    bytes: 1 + 1 / 16384,
    blockSize: 16384,
    note: 'One byte for each weight and a wider exponent range than E4M3. Block scaled like E4M3.',
  },
  {
    id: 'INT8',
    label: 'INT8',
    bytes: 1,
    blockSize: null,
    note: 'One byte for each weight. An integer format, so it needs a scale and a zero point for the whole tensor.',
  },
  {
    id: 'MXFP4',
    label: 'MXFP4',
    bytes: 0.5 + 1 / 32,
    blockSize: 32,
    note: 'Half a byte for each weight plus one shared E8M0 scale for each 32 values. One value costs 4.25 bits.',
  },
  {
    id: 'NVFP4',
    label: 'NVFP4',
    bytes: 0.5 + 1 / 16,
    blockSize: 16,
    note: 'Half a byte for each weight plus one shared E4M3 scale for each 16 values. One value costs 4.5 bits.',
  },
  {
    id: 'INT4',
    label: 'INT4',
    bytes: 0.5 + 2 / 128,
    blockSize: 128,
    note: 'Half a byte for each weight plus a 16 bit scale and zero point for each group of 128 values.',
  },
]

const FORMAT_BY_ID = new Map(WEIGHT_FORMATS.map((format) => [format.id, format]))

/** The format a model is assumed to be in when its config names none. */
export const DEFAULT_WEIGHT_FORMAT: WeightFormatId = 'BF16'

/** Every format id, in the order the tables above list them. */
export const WEIGHT_FORMAT_IDS: WeightFormatId[] = WEIGHT_FORMATS.map((format) => format.id)

/** Narrows an arbitrary string to a known weight format id. */
export function isWeightFormatId(value: string): value is WeightFormatId {
  return FORMAT_BY_ID.has(value as WeightFormatId)
}

export function getWeightFormat(id: WeightFormatId): WeightFormatSpec {
  const spec = FORMAT_BY_ID.get(id)
  if (!spec) throw new Error(`Unknown weight format: ${id}`)
  return spec
}

/** Human readable label for a format. */
export function weightFormatLabel(id: WeightFormatId): string {
  return FORMAT_BY_ID.get(id)?.label ?? id
}

/** True when a format stores 4 bit values, which is what an FP4 tensor path runs. */
export function isFourBitFormat(id: WeightFormatId): boolean {
  return id === 'MXFP4' || id === 'NVFP4' || id === 'INT4'
}

/** True when a format is one of the FP8 widths. */
export function isFp8Format(id: WeightFormatId): boolean {
  return id === 'FP8_E4M3' || id === 'FP8_E5M2'
}

/**
 * Bytes for each weight, including the share of the scale sidecar.
 *
 * An FP8 checkpoint names its own block size, so the caller can pass it. Every
 * other block scaled format has a fixed block, and a format with no scale
 * ignores the argument.
 */
export function bytesPerWeight(
  id: WeightFormatId,
  options: { blockSize?: number | null } = {},
): number {
  const spec = getWeightFormat(id)
  if (isFp8Format(id)) {
    const blockSize = options.blockSize ?? spec.blockSize ?? 16384
    return 1 + 1 / blockSize
  }
  return spec.bytes
}
