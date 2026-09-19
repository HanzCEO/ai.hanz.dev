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
}

export const ROUTE_META: RouteMeta[] = [
  {
    path: '/',
    title: 'ai.hanz.dev | Tools for AI developers',
    description:
      'Small, focused tools for people building with language models. Start with the KV cache calculator, which sizes a cache straight from the model config.',
  },
  {
    path: '/tools/kv-cache-calculator',
    title: 'KV Cache Calculator | ai.hanz.dev',
    description:
      'Estimate KV cache size for any model on HuggingFace or ModelScope. Set context length, sequence count, and cache dtype, then read the full calculation.',
  },
]

const NOT_FOUND_META: RouteMeta = {
  path: '/404',
  title: 'Page not found | ai.hanz.dev',
  description: 'That page does not exist. Head back to the tool list.',
}

/** Routes that get their own prerendered HTML file. */
export const PRERENDER_ROUTES: string[] = ROUTE_META.map((meta) => meta.path)

function normalize(pathname: string): string {
  if (pathname.length > 1) return pathname.replace(/\/+$/, '')
  return pathname
}

export function getRouteMeta(pathname: string): RouteMeta {
  const path = normalize(pathname)
  const match = ROUTE_META.find((meta) => meta.path === path)
  if (match) return match
  return { ...NOT_FOUND_META, path: path === '' ? '/404' : path }
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
  const url = canonicalUrl(meta.path)
  return [
    { tag: 'title', attrs: {}, text: meta.title },
    { tag: 'meta', attrs: { name: 'description', content: meta.description } },
    { tag: 'link', attrs: { rel: 'canonical', href: url } },
    { tag: 'meta', attrs: { property: 'og:type', content: 'website' } },
    { tag: 'meta', attrs: { property: 'og:site_name', content: SITE_NAME } },
    { tag: 'meta', attrs: { property: 'og:title', content: meta.title } },
    { tag: 'meta', attrs: { property: 'og:description', content: meta.description } },
    { tag: 'meta', attrs: { property: 'og:url', content: url } },
    { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary' } },
    { tag: 'meta', attrs: { name: 'twitter:title', content: meta.title } },
    { tag: 'meta', attrs: { name: 'twitter:description', content: meta.description } },
  ]
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
