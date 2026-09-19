import type { ArchitectureFamily, DtypeId, DtypeSpec, DtypeSupport, SupportLevel } from './types'

/**
 * Every dtype the dropdowns offer. Bytes per element is what the size formula
 * actually consumes; fractional values are sub-byte formats that still occupy
 * a whole byte in practice once packed with their scales.
 */
export const DTYPES: DtypeSpec[] = [
  { id: 'BF16', label: 'BF16', bytes: 2 },
  { id: 'FP16', label: 'FP16', bytes: 2 },
  { id: 'FP32', label: 'FP32', bytes: 4 },
  { id: 'FP8_E4M3', label: 'FP8 (E4M3)', bytes: 1 },
  { id: 'FP8_E5M2', label: 'FP8 (E5M2)', bytes: 1 },
  { id: 'INT8', label: 'INT8', bytes: 1 },
  { id: 'INT4', label: 'INT4', bytes: 0.5 },
  { id: 'FP4', label: 'FP4', bytes: 0.5 },
]

const DTYPE_BY_ID = new Map(DTYPES.map((dtype) => [dtype.id, dtype]))

/** Narrows an arbitrary string to a known dtype id. */
export function isDtypeId(value: string): value is DtypeId {
  return DTYPE_BY_ID.has(value as DtypeId)
}

export function getDtype(id: DtypeId): DtypeSpec {
  const spec = DTYPE_BY_ID.get(id)
  if (!spec) throw new Error(`Unknown dtype: ${id}`)
  return spec
}

export function dtypeBytes(id: DtypeId): number {
  return getDtype(id).bytes
}

/** Human readable label for a dtype, falling back to the raw id. */
export function dtypeLabel(id: DtypeId): string {
  return DTYPE_BY_ID.get(id)?.label ?? id
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
    return { level: 'supported', reason: 'The default cache dtype.' }
  }

  if (dtype === 'FP16') {
    if (family === 'mla' || family === 'dsv4' || family === 'dsv41') {
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
    if (family === 'dsv41') {
      if (dtype === 'FP4') {
        return {
          level: 'supported',
          reason: 'V4.1 stores the indexer keys as MXFP4 with one UE8M0 scale per 32 values, the format it was trained with.',
        }
      }
      if (dtype === 'FP8_E4M3') {
        return {
          level: 'untested',
          reason: 'V4.1 moved the indexer cache to FP4. FP8 doubles its footprint and is not the trained format.',
        }
      }
    }
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
      if (family === 'dsv41') {
        return {
          level: 'untested',
          reason: 'V4.1 stores indexer keys as MXFP4, so an FP8 indexer cache is off the trained path.',
        }
      }
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
    if (family === 'dsv41') {
      return {
        level: 'untested',
        reason: 'V4.1 caches the compressed main KV in FP4. FP8 doubles that footprint, though the indexer keys were FP8 in V4.',
      }
    }
    return {
      level: 'supported',
      reason: 'vLLM and SGLang both expose an FP8 (E4M3) KV cache. Verify accuracy first.',
    }
  }

  if (dtype === 'FP8_E5M2') {
    if (family === 'dsv4' || family === 'dsv41') {
      return { level: 'untested', reason: 'The V4 attention kernel targets E4M3.' }
    }
    return { level: 'untested', reason: 'vLLM accepts E5M2, but E4M3 keeps one more mantissa bit, so E4M3 is the usual cache choice.' }
  }

  if (dtype === 'INT8') {
    if (family === 'gqa') {
      return {
        level: 'untested',
        reason: 'Integer caches need extra scale storage, which eats into the memory saving.',
      }
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
    if (family === 'dsv41') {
      return {
        level: 'supported',
        reason: 'V4.1 quantizes the compressed main KV to FP4 (E2M1) with one E4M3 scale per 16 channels, the format it was trained with.',
      }
    }
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
