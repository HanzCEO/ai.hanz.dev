import { Calculator, Cpu, Scissors, Sparkles, type LucideIcon } from 'lucide-react'

export type ToolStatus = 'ready' | 'planned'

export interface Tool {
  slug: string
  name: string
  /** Shown on the card. */
  description: string
  icon: LucideIcon
  status: ToolStatus
}

/**
 * Adding a tool is a one line change here plus a route in src/App.tsx and an
 * entry in ROUTE_META. The landing page grid renders from this array.
 */
export const tools: Tool[] = [
  {
    slug: 'kv-cache-calculator',
    name: 'KV Cache Calculator',
    description:
      'Sizes the cache for models using grouped query attention, MLA, hybrid linear attention, or sparse attention.',
    icon: Calculator,
    status: 'ready',
  },
  {
    slug: 'cost-to-reap-calculator',
    name: 'REAP Cost Calculator',
    description:
      'Estimates how long REAP expert pruning takes for a mixture of experts model, whether one expert block fits your GPU, and how much smaller the pruned model gets.',
    icon: Scissors,
    status: 'ready',
  },
  {
    slug: 'dspark-training-cost-calculator',
    name: 'DSpark Training Cost Calculator',
    description:
      'Estimates the cost of training a DSpark speculative-decoding drafter against a custom target model: the size of the target hidden state cache, the length of the run, and whether it fits in VRAM.',
    icon: Sparkles,
    status: 'ready',
  },
  {
    slug: 'inference-gpu-calculator',
    name: 'Inference GPU Calculator',
    description:
      'Sizes the KV cache for a model and then finds the smallest GPU configuration that serves it in the weight format the checkpoint publishes, from the resident weights, the cache, an activation buffer, and a framework reserve.',
    icon: Cpu,
    status: 'ready',
  },
]

/**
 * The path a tool card links to. It carries the trailing slash because that is
 * the form the host serves with a 200, so a click does not spend a redirect.
 */
export function toolPath(slug: string): string {
  return `/tools/${slug}/`
}
