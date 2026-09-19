import { readFileSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { ROUTE_META, SITE_ORIGIN, canonicalUrl } from './seo'

/**
 * The committed sitemap is what the dev server hands out, and the prerender
 * step rewrites it in dist during a build. Guarding the committed copy keeps a
 * stale file from advertising URLs that disagree with the generated one.
 */
const SITEMAP_PATH = path.join(__dirname, '..', '..', 'public', 'sitemap.xml')
const sitemap = readFileSync(SITEMAP_PATH, 'utf8')

const entries = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1])

describe('public/sitemap.xml', () => {
  it('lists every route once, under its canonical URL', () => {
    expect(entries).toEqual(ROUTE_META.map((meta) => canonicalUrl(meta.path)))
  })

  /**
   * Search Console rejects a sitemap that carries any element the protocol does
   * not define, so the tag set is pinned rather than left to the generator.
   */
  it('uses only tags the sitemap protocol defines', () => {
    const tags = new Set(
      [...sitemap.matchAll(/<\/?([a-zA-Z][\w:-]*)/g)].map((match) => match[1]),
    )
    expect([...tags].sort()).toEqual(['changefreq', 'loc', 'url', 'urlset'])
  })

  it('gives every nested URL the trailing slash the host serves', () => {
    for (const loc of entries) {
      const pathname = loc.slice(SITE_ORIGIN.length)
      if (pathname === '/') continue
      expect(pathname.endsWith('/')).toBe(true)
    }
  })

  it('never lists the not found page', () => {
    expect(sitemap).not.toContain('/404')
  })
})