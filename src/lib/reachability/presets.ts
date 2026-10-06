/**
 * Defaults and groupings for the reachability leaderboard.
 *
 * Every number here is a stated allowance rather than a measurement, so the
 * page can repeat it and a reader can change it. The sizing itself is done by
 * the inference engine in src/lib/inference, which the inference GPU
 * calculator already uses. This module only decides what workload that engine
 * is asked about.
 */

import { GPU_PRESETS } from '../hardware'
import { DEFAULT_HEADROOM, MAX_SUGGESTED_GPUS } from '../inference'
import type { DtypeId } from '../kvcache'
import type { WeightFormatId } from '../weight-format'

/**
 * The context length the page opens on.
 *
 * A classification or routing call is short. Four thousand and ninety six
 * tokens is well above a typical prompt, so a model that fits here fits the
 * job the leaderboard is about.
 */
export const DEFAULT_REACHABILITY_CONTEXT = 4096

/** The sequences served at once. One request is the conservative default. */
export const DEFAULT_REACHABILITY_SEQUENCES = 1

/**
 * The weight format the page opens on.
 *
 * Null means the format the checkpoint publishes, which is what the model was
 * trained and released in. A reader can force a narrower format to see which
 * cards a quantised copy would reach.
 */
export const DEFAULT_REACHABILITY_WEIGHT_FORMAT: WeightFormatId | null = null

/** The card limit the ranking uses before it reports that nothing fits. */
export const DEFAULT_REACHABILITY_MAX_GPUS = MAX_SUGGESTED_GPUS

/** The share of each card held back for fragmentation and the display. */
export const DEFAULT_REACHABILITY_HEADROOM = DEFAULT_HEADROOM

/** The dtype the KV cache is costed in. */
export const DEFAULT_REACHABILITY_KV_DTYPE: DtypeId = 'BF16'

/** A hardware class, so a reader can ask what a tier of card can reach. */
export type ReachabilityGpuClass = 'consumer' | 'workstation' | 'datacenter'

/**
 * The datacenter cards in the hardware directory.
 *
 * The classification is named rather than derived from the memory size,
 * because an 80 GB H100 is a datacenter card while a 96 GB RTX PRO 6000 is a
 * workstation one. Memory alone would put them the wrong way round.
 */
const DATACENTER_GPU_IDS = ['b300', 'b200', 'h200', 'h100', 'mi300x']

/** The workstation cards in the hardware directory. */
const WORKSTATION_GPU_IDS = ['rtx-pro-6000']

/** Every id named above, so the lists can be checked against the directory. */
const NAMED_GPU_IDS = new Set([...DATACENTER_GPU_IDS, ...WORKSTATION_GPU_IDS])

// A named id that is not in the directory would silently narrow the filter, so
// the mistake is raised at module load rather than hidden in a wrong answer.
for (const id of NAMED_GPU_IDS) {
  if (!GPU_PRESETS.some((gpu) => gpu.id === id)) {
    throw new Error(`The reachability grouping names an unknown GPU: ${id}`)
  }
}

export interface ReachabilityGpuGroup {
  id: ReachabilityGpuClass
  label: string
  hint: string
  gpuIds: string[]
}

/**
 * The three hardware classes, widest reach last.
 *
 * A card that is not named as datacenter or workstation is a consumer card.
 * That is the default for a new entry in the directory, because the consumer
 * list is the largest group.
 */
export const REACHABILITY_GPU_GROUPS: ReachabilityGpuGroup[] = [
  {
    id: 'consumer',
    label: 'Consumer',
    hint: 'Desktop cards from 8 GB to 32 GB.',
    gpuIds: GPU_PRESETS.filter((gpu) => !NAMED_GPU_IDS.has(gpu.id)).map((gpu) => gpu.id),
  },
  {
    id: 'workstation',
    label: 'Workstation',
    hint: 'The RTX PRO 6000, the largest single workstation card here.',
    gpuIds: GPU_PRESETS.filter((gpu) => WORKSTATION_GPU_IDS.includes(gpu.id)).map(
      (gpu) => gpu.id,
    ),
  },
  {
    id: 'datacenter',
    label: 'Datacenter',
    hint: 'The B300, B200, H200, H100, and MI300X.',
    gpuIds: GPU_PRESETS.filter((gpu) => DATACENTER_GPU_IDS.includes(gpu.id)).map((gpu) => gpu.id),
  },
]

/** The classes the picker offers, with the all cards option in front. */
export type ReachabilityGpuClassFilter = ReachabilityGpuClass | 'all'

/** The vendor filter, with the all vendors option in front. */
export type ReachabilityVendorFilter = 'all' | 'nvidia' | 'amd'

/**
 * The ids a class and vendor filter allows, in directory order.
 *
 * The filter is expressed as two small choices rather than a forty row list,
 * because a reader asking "what runs on a consumer card" does not want to tick
 * thirty boxes. The engine still receives a plain id list, so a caller that
 * wants a different selection can pass one.
 */
export function gpuIdsForFilter(
  gpuClass: ReachabilityGpuClassFilter,
  vendor: ReachabilityVendorFilter,
): string[] {
  return GPU_PRESETS.filter((gpu) => {
    if (vendor !== 'all' && gpu.vendor !== vendor) return false
    if (gpuClass === 'all') return true
    const group = REACHABILITY_GPU_GROUPS.find((entry) => entry.id === gpuClass)
    return group ? group.gpuIds.includes(gpu.id) : false
  }).map((gpu) => gpu.id)
}

/** How the table is ordered. */
export type ReachabilitySortKey = 'reachability' | 'index' | 'decode' | 'weights' | 'name'

export interface ReachabilitySort {
  id: ReachabilitySortKey
  label: string
  hint: string
}

/** The sort orders the picker offers, the default first. */
export const REACHABILITY_SORTS: ReachabilitySort[] = [
  {
    id: 'reachability',
    label: 'Reachability',
    hint: 'The smallest card that holds the model first, then the models no card holds.',
  },
  {
    id: 'index',
    label: 'Decision Index',
    hint: 'The highest Decision Index first, at any hardware.',
  },
  {
    id: 'decode',
    label: 'Decode rate',
    hint: 'The fastest recommended configuration first.',
  },
  {
    id: 'weights',
    label: 'Checkpoint size',
    hint: 'The smallest checkpoint first, which is the easiest to host.',
  },
  {
    id: 'name',
    label: 'Name',
    hint: 'Alphabetical by model name.',
  },
]

/** The sort the page opens on. */
export const DEFAULT_REACHABILITY_SORT: ReachabilitySortKey = 'reachability'
