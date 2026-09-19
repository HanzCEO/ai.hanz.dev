import { Calculator, type LucideIcon } from 'lucide-react'

export type ToolStatus = 'ready' | 'planned'

export interface Tool {
  slug: string
  name: string
  /** One line, shown on the card. */
  tagline: string
  /** A short paragraph, shown on the card. */
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
    tagline: 'How much memory the cache will actually take',
    description:
      'Sizes the cache from the real model config, not a hardcoded table. Covers grouped query attention, MLA, hybrid linear attention, and sparse attention.',
    icon: Calculator,
    status: 'ready',
  },
]

export function toolPath(slug: string): string {
  return `/tools/${slug}`
}
