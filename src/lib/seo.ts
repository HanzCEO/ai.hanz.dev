import { REAP_FAQ } from './reap/faq'

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
  /**
   * Structured data written into a JSON-LD script. Built with faqJsonLd,
   * softwareApplicationJsonLd, or jsonLdGraph, so a page can describe itself to
   * a search engine in a form it can read directly.
   */
  jsonLd?: Record<string, unknown>
}

/**
 * Route paths carry a trailing slash because that is the only form GitHub Pages
 * answers with a 200. A request for the bare form is a 301 to the slashed form,
 * so any canonical, og:url, sitemap entry, or internal link built from the bare
 * form would advertise a redirect instead of the page itself.
 */
export const KV_CACHE_PATH = '/tools/kv-cache-calculator/'
export const REAP_PATH = '/tools/reap-cost-calculator/'

const REAP_TITLE = 'REAP Duration Calculator | ai.hanz.dev'
const REAP_DESCRIPTION =
  'REAP duration calculator for mixture of experts models. Estimate how long REAP expert pruning takes, check whether one expert block fits your GPU, and see how much smaller the pruned model gets.'

export const ROUTE_META: RouteMeta[] = [
  {
    path: '/',
    title: 'ai.hanz.dev | Tools for AI developers',
    description:
      'Tools for AI developers. Size a KV cache from a model config, or estimate the cost of pruning a mixture of experts with REAP.',
  },
  {
    path: KV_CACHE_PATH,
    title: 'KV Cache Calculator | ai.hanz.dev',
    description:
      'Estimate KV cache size for any model on HuggingFace or ModelScope. Set context length, sequence count, and cache dtype.',
  },
  {
    path: REAP_PATH,
    title: REAP_TITLE,
    description: REAP_DESCRIPTION,
    // The visible questions and the markup a search engine reads come from one
    // list, so they cannot drift apart.
    jsonLd: jsonLdGraph(
      softwareApplicationJsonLd({
        path: REAP_PATH,
        title: REAP_TITLE,
        description: REAP_DESCRIPTION,
      }),
      faqJsonLd(REAP_FAQ),
    ),
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

/**
 * Returns the matching route, or null when nothing matches.
 *
 * Both sides are normalized because the route path carries the canonical slash
 * while a request may or may not. That keeps /a and /a/ resolving to the same
 * entry even on a host that does not redirect the bare form.
 */
export function matchRoute(pathname: string): RouteMeta | null {
  const path = normalizePath(pathname)
  return ROUTE_META.find((meta) => normalizePath(meta.path) === path) ?? null
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
 *
 * The answer is the route's own canonical path, because that is the string the
 * prerender step stamps onto #root. Returning the normalized request path
 * instead would hand back the bare form for a slashed request and make every
 * hydration check fail.
 */
export function prerenderMarkerFor(pathname: string): string {
  return matchRoute(pathname)?.path ?? PRERENDER_WILDCARD
}

export function canonicalUrl(path: string): string {
  return `${SITE_ORIGIN}${path}`
}

export interface HeadTag {
  tag: 'title' | 'meta' | 'link' | 'script'
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

  if (meta.jsonLd) {
    tags.push({
      tag: 'script',
      attrs: { type: 'application/ld+json' },
      text: JSON.stringify(meta.jsonLd),
    })
  }

  return tags
}

/**
 * Escapes a JSON body so it cannot close the script element it sits in.
 *
 * A payload containing the literal text of a closing script tag would end the
 * element early and turn the rest of the document into markup. Replacing the
 * opening angle bracket with its unicode escape is valid inside a JSON string
 * and leaves the parsed value unchanged.
 */
function escapeJsonForScript(json: string): string {
  return json.replace(/</g, '\\u003c')
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
      if (tag.tag === 'script') {
        return `<script${SEO_MARKER}${attrs}>${escapeJsonForScript(tag.text ?? '')}</script>`
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
    // A JSON-LD body is assigned as text content, never parsed as markup.
    if (tag.text !== undefined) element.textContent = tag.text
    element.setAttribute(SEO_ATTR, '')
    document.head.appendChild(element)
  }
}

// ---------------------------------------------------------------------------
// Structured data
// ---------------------------------------------------------------------------

export interface FaqItem {
  question: string
  answer: string
}

/**
 * A FAQPage node. Search engines read these question and answer pairs directly,
 * and they are the shape an AI overview lifts its answer from.
 */
export function faqJsonLd(items: FaqItem[]): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: items.map((item) => ({
      '@type': 'Question',
      name: item.question,
      acceptedAnswer: { '@type': 'Answer', text: item.answer },
    })),
  }
}

/** A SoftwareApplication node describing the tool on a page. */
export function softwareApplicationJsonLd(meta: RouteMeta): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: meta.title,
    description: meta.description,
    url: canonicalUrl(meta.path),
    applicationCategory: 'DeveloperApplication',
    operatingSystem: 'Any',
    isAccessibleForFree: true,
    offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
    publisher: { '@type': 'Organization', name: SITE_NAME, url: SITE_ORIGIN },
  }
}

/**
 * Combines several structured data nodes into one document. A single script can
 * carry a graph, so a page needs only one JSON-LD block. The per node context
 * is dropped because the graph declares it once.
 */
export function jsonLdGraph(...nodes: Array<Record<string, unknown>>): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@graph': nodes.map((node) => {
      const copy = { ...node }
      delete copy['@context']
      return copy
    }),
  }
}
