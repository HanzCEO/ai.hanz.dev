import type { ArchitectureFamily, DtypeId, DtypeSpec, DtypeSupport, SupportLevel } from './types'

/**
 * Every dtype the dropdowns offer. Bytes per element is what the size formula
 * actually consumes; fractional values are sub-byte formats that still occupy
 * a whole byte in practice once packed with their scales.
 */
export const DTYPES: DtypeSpec[] = [
  { id: 'BF16', label: 'BF16', bytes: 2, note: 'Brain float. Default for most caches.' },
  { id: 'FP16', label: 'FP16', bytes: 2, note: 'Same width as BF16, narrower exponent range.' },
  {
    id: 'FP32',
    label: 'FP32',
    bytes: 4,
    note: 'Unquantized. Doubles the cache for no accuracy you actually need.',
  },
  {
    id: 'FP8_E4M3',
    label: 'FP8 (E4M3)',
    bytes: 1,
    note: 'Most common quantized cache. Needs an accuracy check.',
  },
  {
    id: 'FP8_E5M2',
    label: 'FP8 (E5M2)',
    bytes: 1,
    note: 'Wider exponent range, fewer mantissa bits.',
  },
  { id: 'INT8', label: 'INT8', bytes: 1, note: 'Integer cache with per-channel scales.' },
  { id: 'INT4', label: 'INT4', bytes: 0.5, note: 'Packed 4 bit with a scale per group.' },
  {
    id: 'FP4',
    label: 'FP4',
    bytes: 0.5,
    note: '4 bit float. Used for indexer caches in production.',
  },
]

const DTYPE_BY_ID = new Map(DTYPES.map((dtype) => [dtype.id, dtype]))

export function getDtype(id: DtypeId): DtypeSpec {
  const spec = DTYPE_BY_ID.get(id)
  if (!spec) throw new Error(`Unknown dtype: ${id}`)
  return spec
}

export function dtypeBytes(id: DtypeId): number {
  return getDtype(id).bytes
}

export const DEFAULT_DTYPE: DtypeId = 'BF16'

export type DtypeRole = 'kv' | 'indexer'

type Rule = { level: SupportLevel; reason: string }

/**
 * Grounds each tag in how the architecture is actually served. Nothing is
 * marked supported without a real basis, and the UI shows the reason next to
 * the option so the tag is not a bare assertion.
 */
function ruleFor(family: ArchitectureFamily, role: DtypeRole, dtype: DtypeId): Rule {
  if (dtype === 'BF16') {
    return { level: 'supported', reason: 'The default cache dtype nearly everywhere.' }
  }

  if (dtype === 'FP16') {
    if (family === 'mla' || family === 'dsv4') {
      return {
        level: 'untested',
        reason: 'MLA kernels are tuned for BF16. FP16 works in theory but is rarely exercised.',
      }
    }
    return { level: 'supported', reason: 'Interchangeable with BF16 for cache storage.' }
  }

  if (dtype === 'FP32') {
    return {
      level: 'unsupported',
      reason: 'No mainstream serving engine exposes an FP32 cache, and it doubles the size.',
    }
  }

  if (role === 'indexer') {
    if (family === 'dsv4') {
      if (dtype === 'FP4') {
        return {
          level: 'supported',
          reason: 'Production V4 deployments run the indexer cache in FP4.',
        }
      }
      if (dtype === 'FP8_E4M3') {
        return { level: 'supported', reason: 'A supported indexer cache format for V4.' }
      }
      if (dtype === 'FP8_E5M2') {
        return { level: 'untested', reason: 'The V4 indexer path is built around E4M3.' }
      }
    }
    if (family === 'mla') {
      if (dtype === 'FP8_E4M3') {
        return {
          level: 'supported',
          reason: 'The DeepSeek-V3.2 indexer stores block-scaled FP8 keys.',
        }
      }
      if (dtype === 'FP4') {
        return { level: 'untested', reason: 'Not used for this indexer in current deployments.' }
      }
    }
    if (dtype === 'FP8_E4M3' || dtype === 'FP8_E5M2') {
      return { level: 'untested', reason: 'No indexer exists for this architecture family.' }
    }
    if (dtype === 'INT8' || dtype === 'INT4') {
      return { level: 'unsupported', reason: 'Indexer caches are not stored as integers.' }
    }
    if (dtype === 'FP4') {
      return { level: 'untested', reason: 'Not used for this indexer in current deployments.' }
    }
  }

  // KV cache role.
  if (dtype === 'FP8_E4M3') {
    if (family === 'hybrid_linear') {
      return {
        level: 'untested',
        reason: 'Linear attention state is kept in higher precision, so FP8 is unusual here.',
      }
    }
    return {
      level: 'supported',
      reason: 'vLLM and SGLang both expose an FP8 (E4M3) KV cache. Verify accuracy first.',
    }
  }

  if (dtype === 'FP8_E5M2') {
    if (family === 'dsv4') {
      return { level: 'untested', reason: 'The V4 attention kernel targets E4M3.' }
    }
    return { level: 'untested', reason: 'Supported by some engines, less commonly deployed.' }
  }

  if (dtype === 'INT8') {
    if (family === 'gqa') {
      return { level: 'untested', reason: 'Available in some engines, not a common default.' }
    }
    return { level: 'unsupported', reason: 'Not supported by the kernels for this architecture.' }
  }

  if (dtype === 'INT4') {
    if (family === 'gqa') {
      return { level: 'untested', reason: 'Experimental in a few engines. Expect accuracy loss.' }
    }
    return { level: 'unsupported', reason: 'Not supported by the kernels for this architecture.' }
  }

  if (dtype === 'FP4') {
    if (family === 'dsv4') {
      return {
        level: 'untested',
        reason: 'V4 uses FP4 for the indexer cache. The attention cache stays FP8.',
      }
    }
    return { level: 'unsupported', reason: 'Not supported for the attention cache.' }
  }

  return { level: 'untested', reason: 'No data for this combination.' }
}

export function dtypeSupport(family: ArchitectureFamily, role: DtypeRole): DtypeSupport[] {
  return DTYPES.map((dtype) => {
    const rule = ruleFor(family, role, dtype.id)
    return { dtype: dtype.id, level: rule.level, reason: rule.reason }
  })
}

export function supportFor(
  family: ArchitectureFamily,
  role: DtypeRole,
  dtype: DtypeId,
): DtypeSupport {
  const found = dtypeSupport(family, role).find((entry) => entry.dtype === dtype)
  return found ?? { dtype, level: 'untested', reason: 'No data for this combination.' }
}
