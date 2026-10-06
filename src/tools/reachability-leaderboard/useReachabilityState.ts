import { useMemo } from 'react'

import { DTYPES, type DtypeId } from '@/lib/kvcache'
import {
  DEFAULT_REACHABILITY_CONTEXT,
  DEFAULT_REACHABILITY_HEADROOM,
  DEFAULT_REACHABILITY_KV_DTYPE,
  DEFAULT_REACHABILITY_MAX_GPUS,
  DEFAULT_REACHABILITY_SEQUENCES,
  DEFAULT_REACHABILITY_SORT,
  REACHABILITY_SORTS,
  gpuIdsForFilter,
  rankReachability,
  type ReachabilityGpuClassFilter,
  type ReachabilityRow,
  type ReachabilitySortKey,
  type ReachabilityVendorFilter,
} from '@/lib/reachability'
import {
  digitsOrNull,
  enumOf,
  nonEmptyText,
  parsePositiveInteger,
  useUrlSyncedState,
  type UrlSchema,
} from '@/lib/url-state'
import {
  WEIGHT_FORMAT_IDS,
  isWeightFormatId,
  weightFormatLabel,
  type WeightFormatId,
} from '@/lib/weight-format'

/**
 * The weight format the table costs every checkpoint in.
 *
 * `published` follows the format each checkpoint was released in, which is the
 * default because it is what a reader would download. Every other value forces
 * that one format on the whole model, which answers the question of what a
 * quantised copy would reach.
 */
export type ReachabilityWeightChoice = 'published' | WeightFormatId

/** The filter fields, plus the workload and the sort order. */
export interface ReachabilityFormInputs {
  /** Free text matched against the model name, engine, publisher, and base model. */
  search: string
  gpuClass: ReachabilityGpuClassFilter
  vendor: ReachabilityVendorFilter
  contextLength: string
  sequences: string
  weightFormat: ReachabilityWeightChoice
  maxGpus: string
  kvCacheDtype: DtypeId
  sort: ReachabilitySortKey
}

const GPU_CLASSES: ReachabilityGpuClassFilter[] = [
  'all',
  'consumer',
  'workstation',
  'datacenter',
]
const VENDORS: ReachabilityVendorFilter[] = ['all', 'nvidia', 'amd']
const DTYPE_IDS: DtypeId[] = DTYPES.map((dtype) => dtype.id)
const SORT_IDS: ReachabilitySortKey[] = REACHABILITY_SORTS.map((sort) => sort.id)

/** Reads the published choice or a known format, and nothing else. */
function parseWeightChoice(raw: string | null): ReachabilityWeightChoice | null {
  if (raw === null) return null
  if (raw === 'published') return 'published'
  return isWeightFormatId(raw) ? raw : null
}

/**
 * The query string fields the page owns.
 *
 * Declared at module scope, because the URL hook captures the schema in a ref
 * and a schema rebuilt on each render would restart its effects.
 */
export const REACHABILITY_SCHEMA: UrlSchema<ReachabilityFormInputs> = {
  search: { param: 'q', default: '', parse: nonEmptyText },
  gpuClass: { param: 'class', default: 'all', parse: enumOf(GPU_CLASSES) },
  vendor: { param: 'vendor', default: 'all', parse: enumOf(VENDORS) },
  contextLength: {
    param: 'ctx',
    default: String(DEFAULT_REACHABILITY_CONTEXT),
    parse: digitsOrNull,
  },
  sequences: {
    param: 'seq',
    default: String(DEFAULT_REACHABILITY_SEQUENCES),
    parse: digitsOrNull,
  },
  weightFormat: { param: 'weights', default: 'published', parse: parseWeightChoice },
  maxGpus: {
    param: 'max_gpus',
    default: String(DEFAULT_REACHABILITY_MAX_GPUS),
    parse: digitsOrNull,
  },
  kvCacheDtype: { param: 'kv_dtype', default: DEFAULT_REACHABILITY_KV_DTYPE, parse: enumOf(DTYPE_IDS) },
  sort: { param: 'sort', default: DEFAULT_REACHABILITY_SORT, parse: enumOf(SORT_IDS) },
}

/** The validation messages, one per numeric field. */
export interface ReachabilityErrors {
  contextLength: string | null
  sequences: string | null
  maxGpus: string | null
}

/** Everything the form needs to mark a field, keyed by the field name. */
function validate(inputs: ReachabilityFormInputs): ReachabilityErrors {
  return {
    contextLength:
      parsePositiveInteger(inputs.contextLength) === null
        ? 'Enter a whole number of tokens, 1 or more.'
        : null,
    sequences:
      parsePositiveInteger(inputs.sequences) === null
        ? 'Enter a whole number of sequences, 1 or more.'
        : null,
    maxGpus:
      parsePositiveInteger(inputs.maxGpus) === null
        ? 'Enter a whole number of cards, 1 or more.'
        : null,
  }
}

/** The text a search matches against, folded to lower case once. */
function haystack(row: ReachabilityRow): string {
  const model = row.model
  return [model.name, model.engine, model.org ?? '', model.kind, model.baseModel ?? '']
    .join(' ')
    .toLowerCase()
}

/** Applies a sort order on top of the reachability order the engine returned. */
export function sortRows(rows: ReachabilityRow[], key: ReachabilitySortKey): ReachabilityRow[] {
  if (key === 'reachability') return rows
  const sorted = [...rows]
  if (key === 'index') {
    sorted.sort((a, b) => (b.model.index ?? -Infinity) - (a.model.index ?? -Infinity))
  } else if (key === 'decode') {
    sorted.sort((a, b) => b.decodeTokensPerSecond - a.decodeTokensPerSecond)
  } else if (key === 'weights') {
    // A row that could not be sized has no footprint, so it goes last rather
    // than first, which is where a zero would otherwise put it.
    sorted.sort((a, b) => {
      if (!a.sizeable && !b.sizeable) return 0
      if (!a.sizeable) return 1
      if (!b.sizeable) return -1
      return a.weightsBytes - b.weightsBytes
    })
  } else {
    sorted.sort((a, b) => a.model.name.localeCompare(b.model.name))
  }
  return sorted
}

export function useReachabilityState() {
  const { values: inputs, update, serialized } =
    useUrlSyncedState<ReachabilityFormInputs>(REACHABILITY_SCHEMA)

  const errors = useMemo(() => validate(inputs), [inputs])
  const valid = errors.contextLength === null && errors.sequences === null && errors.maxGpus === null

  const gpuIds = useMemo(
    () => gpuIdsForFilter(inputs.gpuClass, inputs.vendor),
    [inputs.gpuClass, inputs.vendor],
  )

  const weightFormat: WeightFormatId | null =
    inputs.weightFormat === 'published' ? null : inputs.weightFormat

  const contextLength = parsePositiveInteger(inputs.contextLength)
  const sequences = parsePositiveInteger(inputs.sequences)
  const maxGpus = parsePositiveInteger(inputs.maxGpus)

  const rows = useMemo(() => {
    if (contextLength === null || sequences === null || maxGpus === null) return []
    if (gpuIds.length === 0) return []
    return rankReachability({
      contextLength,
      sequences,
      weightFormat,
      maxGpus,
      headroom: DEFAULT_REACHABILITY_HEADROOM,
      kvCacheDtype: inputs.kvCacheDtype,
      gpuIds,
    })
  }, [
    contextLength,
    sequences,
    maxGpus,
    weightFormat,
    inputs.kvCacheDtype,
    gpuIds,
  ])

  const visibleRows = useMemo(() => {
    const query = inputs.search.trim().toLowerCase()
    const matched = query === '' ? rows : rows.filter((row) => haystack(row).includes(query))
    return sortRows(matched, inputs.sort)
  }, [rows, inputs.search, inputs.sort])

  return {
    inputs,
    update,
    errors,
    /** False when a numeric field is not a usable value. */
    valid,
    /** True when the class and vendor filter select no card at all. */
    noGpuMatches: gpuIds.length === 0,
    /** Every row for the workload, before the search box narrows it. */
    totalRows: rows.length,
    rows: visibleRows,
    serialized,
  }
}

/** The format choices the picker offers, the published one first. */
export const REACHABILITY_WEIGHT_OPTIONS: Array<{
  id: ReachabilityWeightChoice
  label: string
}> = [
  { id: 'published', label: 'Published (from each checkpoint)' },
  ...WEIGHT_FORMAT_IDS.map((id) => ({
    id: id as ReachabilityWeightChoice,
    label: weightFormatLabel(id),
  })),
]
