/**
 * Route metadata is the single source of truth for page titles, descriptions,
 * canonical URLs, and the prerender route list. The prerender script and the
 * client both read from here, so a page can never ship with head tags that
 * disagree with its own copy.
 */

export const SITE_ORIGIN = 'https://ai.hanz.dev'
export const SITE_NAME = 'ai.hanz.dev'
export const REPO_URL = 'https://github.com/HanzCEO/ai.hanz.dev'
export const SUPPORT_URL = 'https://discord.gg/Awprm4X4yA'

export interface RouteMeta {
  path: string
  title: string
  description: string
  /** Set on pages that must not be indexed, such as the not found page. */
  noindex?: boolean
}

export const ROUTE_META: RouteMeta[] = [
  {
    path: '/',
    title: 'ai.hanz.dev | Tools for AI developers',
    description:
      'Tools for AI developers. Start with the KV cache calculator, which sizes a cache from the model config.',
  },
  {
    path: '/tools/kv-cache-calculator',
    title: 'KV Cache Calculator | ai.hanz.dev',
    description:
      'Estimate KV cache size for any model on HuggingFace or ModelScope. Set context length, sequence count, and cache dtype.',
  },
]

/** Path the not found page is rendered and marked under. */
export const NOT_FOUND_PATH = '/404'

/**
 * Marker written into the prerendered HTML for the not found page. That file is
 * served for every unknown path, so it cannot claim a specific route.
 */
export const PRERENDER_WILDCARD = '*'

/** Attribute the prerender step writes onto #root to record which route it rendered. */
export const PRERENDER_ATTR = 'data-prerender-route'

/** Routes that get their own prerendered HTML file. */
export const PRERENDER_ROUTES: string[] = ROUTE_META.map((meta) => meta.path)

/** Trailing slashes are not significant, so /a and /a/ are the same route. */
export function normalizePath(pathname: string): string {
  const withoutQuery = pathname.split('?')[0].split('#')[0]
  if (withoutQuery.length > 1) return withoutQuery.replace(/\/+$/, '')
  return withoutQuery
}

/** Returns the matching route, or null when nothing matches. */
export function matchRoute(pathname: string): RouteMeta | null {
  const path = normalizePath(pathname)
  return ROUTE_META.find((meta) => meta.path === path) ?? null
}

export function getNotFoundMeta(): RouteMeta {
  return {
    path: NOT_FOUND_PATH,
    title: 'Page not found | ai.hanz.dev',
    description: 'Head back to the tool list.',
    noindex: true,
  }
}

export function getRouteMeta(pathname: string): RouteMeta {
  return matchRoute(pathname) ?? getNotFoundMeta()
}

/**
 * The marker a correctly prerendered document would carry for this URL. The
 * client compares it against what the server actually sent, so it only hydrates
 * markup that belongs to the route it is rendering.
 */
export function prerenderMarkerFor(pathname: string): string {
  return matchRoute(pathname) ? normalizePath(pathname) : PRERENDER_WILDCARD
}

export function canonicalUrl(path: string): string {
  return `${SITE_ORIGIN}${path}`
}

export interface HeadTag {
  tag: 'title' | 'meta' | 'link'
  attrs: Record<string, string>
  text?: string
}

const SEO_ATTR = 'data-seo'
const SEO_MARKER = ` ${SEO_ATTR}=""`

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function headTags(meta: RouteMeta): HeadTag[] {
  const tags: HeadTag[] = [
    { tag: 'title', attrs: {}, text: meta.title },
    { tag: 'meta', attrs: { name: 'description', content: meta.description } },
  ]

  // A shared not found document is served for arbitrary URLs, so it must not
  // claim a canonical URL, and it must not be indexed.
  if (meta.noindex) {
    tags.push({ tag: 'meta', attrs: { name: 'robots', content: 'noindex, follow' } })
  } else {
    const url = canonicalUrl(meta.path)
    tags.push(
      { tag: 'link', attrs: { rel: 'canonical', href: url } },
      { tag: 'meta', attrs: { property: 'og:type', content: 'website' } },
      { tag: 'meta', attrs: { property: 'og:site_name', content: SITE_NAME } },
      { tag: 'meta', attrs: { property: 'og:title', content: meta.title } },
      { tag: 'meta', attrs: { property: 'og:description', content: meta.description } },
      { tag: 'meta', attrs: { property: 'og:url', content: url } },
    )
  }

  tags.push(
    { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary' } },
    { tag: 'meta', attrs: { name: 'twitter:title', content: meta.title } },
    { tag: 'meta', attrs: { name: 'twitter:description', content: meta.description } },
  )

  return tags
}

/** Serialize head tags into HTML for the prerender step. */
export function renderHeadTags(meta: RouteMeta): string {
  return headTags(meta)
    .map((tag) => {
      const attrs = Object.entries(tag.attrs)
        .map(([key, value]) => ` ${key}="${escapeHtml(value)}"`)
        .join('')
      if (tag.tag === 'title') {
        return `<title${SEO_MARKER}>${escapeHtml(tag.text ?? '')}</title>`
      }
      return `<${tag.tag}${SEO_MARKER}${attrs} />`
    })
    .join('\n    ')
}

/**
 * Client-side equivalent of renderHeadTags. Tags written here carry the same
 * marker as the prerendered ones, so navigation replaces them cleanly instead
 * of stacking duplicates.
 */
export function applyHead(meta: RouteMeta): void {
  if (typeof document === 'undefined') return

  for (const stale of Array.from(document.head.querySelectorAll(`[${SEO_ATTR}]`))) {
    stale.remove()
  }

  for (const tag of headTags(meta)) {
    const element = document.createElement(tag.tag)
    for (const [key, value] of Object.entries(tag.attrs)) {
      element.setAttribute(key, value)
    }
    if (tag.text !== undefined) element.textContent = tag.text
    element.setAttribute(SEO_ATTR, '')
    document.head.appendChild(element)
  }
}
