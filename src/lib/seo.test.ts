import { describe, expect, it } from 'vitest'

import {
  NOT_FOUND_PATH,
  PRERENDER_ROUTES,
  PRERENDER_WILDCARD,
  SITE_ORIGIN,
  faqJsonLd,
  getNotFoundMeta,
  getRouteMeta,
  headTags,
  jsonLdGraph,
  matchRoute,
  normalizePath,
  prerenderMarkerFor,
  renderHeadTags,
  softwareApplicationJsonLd,
} from './seo'

describe('normalizePath', () => {
  it('treats a trailing slash as insignificant', () => {
    expect(normalizePath('/tools/kv-cache-calculator/')).toBe('/tools/kv-cache-calculator')
    expect(normalizePath('/tools/kv-cache-calculator//')).toBe('/tools/kv-cache-calculator')
  })

  it('keeps the root path as a single slash', () => {
    expect(normalizePath('/')).toBe('/')
  })

  it('drops a query string and a hash', () => {
    expect(normalizePath('/tools/kv-cache-calculator?model=Qwen/Qwen3-8B')).toBe(
      '/tools/kv-cache-calculator',
    )
    expect(normalizePath('/tools/kv-cache-calculator#top')).toBe('/tools/kv-cache-calculator')
  })
})

describe('matchRoute', () => {
  it('matches a known route with or without a trailing slash', () => {
    expect(matchRoute('/')?.path).toBe('/')
    expect(matchRoute('/tools/kv-cache-calculator')?.path).toBe('/tools/kv-cache-calculator')
    expect(matchRoute('/tools/kv-cache-calculator/')?.path).toBe('/tools/kv-cache-calculator')
  })

  it('returns null for anything unknown', () => {
    expect(matchRoute('/some/missing/page')).toBeNull()
    expect(matchRoute('/tools')).toBeNull()
    expect(matchRoute('/tools/kv-cache-calculator/extra')).toBeNull()
  })
})

describe('getRouteMeta', () => {
  it('returns the matching route metadata', () => {
    const meta = getRouteMeta('/tools/kv-cache-calculator')
    expect(meta.title).toBe('KV Cache Calculator | ai.hanz.dev')
    expect(meta.noindex).toBeUndefined()
  })

  it('falls back to the not found metadata for an unknown path', () => {
    const meta = getRouteMeta('/nope')
    expect(meta).toEqual(getNotFoundMeta())
    expect(meta.noindex).toBe(true)
    // A shared not found document must not claim the path it was served for.
    expect(meta.path).toBe(NOT_FOUND_PATH)
  })
})

describe('prerenderMarkerFor', () => {
  it('returns the route itself for a known path', () => {
    expect(prerenderMarkerFor('/')).toBe('/')
    expect(prerenderMarkerFor('/tools/kv-cache-calculator')).toBe('/tools/kv-cache-calculator')
    expect(prerenderMarkerFor('/tools/kv-cache-calculator/')).toBe('/tools/kv-cache-calculator')
  })

  it('returns the wildcard for an unknown path, since 404.html covers all of them', () => {
    expect(prerenderMarkerFor('/some/missing/page')).toBe(PRERENDER_WILDCARD)
    expect(prerenderMarkerFor('/another/missing/one')).toBe(PRERENDER_WILDCARD)
  })

  it('never returns a marker that no prerendered document carries', () => {
    const markers = new Set([...PRERENDER_ROUTES, PRERENDER_WILDCARD])
    for (const path of ['/', '/tools/kv-cache-calculator/', '/nope', '/a/b/c']) {
      expect(markers.has(prerenderMarkerFor(path))).toBe(true)
    }
  })
})

describe('headTags', () => {
  it('gives an indexable page a canonical URL and social tags', () => {
    const tags = headTags(getRouteMeta('/tools/kv-cache-calculator'))
    const byName = (name: string) =>
      tags.find((tag) => tag.attrs.rel === name || tag.attrs.property === name || tag.attrs.name === name)

    expect(byName('canonical')?.attrs.href).toBe(
      `${SITE_ORIGIN}/tools/kv-cache-calculator`,
    )
    expect(byName('og:title')?.attrs.content).toBe('KV Cache Calculator | ai.hanz.dev')
    expect(byName('twitter:card')?.attrs.content).toBe('summary')
    expect(tags.find((tag) => tag.attrs.name === 'robots')).toBeUndefined()
  })

  it('marks the not found page noindex and gives it no canonical', () => {
    const tags = headTags(getNotFoundMeta())

    expect(tags.find((tag) => tag.attrs.name === 'robots')?.attrs.content).toBe('noindex, follow')
    expect(tags.find((tag) => tag.attrs.rel === 'canonical')).toBeUndefined()
    // The not found document is served for arbitrary URLs, so an og:url would lie.
    expect(tags.find((tag) => tag.attrs.property === 'og:url')).toBeUndefined()
  })

  it('always emits a title and a description', () => {
    for (const path of ['/', '/tools/kv-cache-calculator', '/unknown']) {
      const tags = headTags(getRouteMeta(path))
      expect(tags.find((tag) => tag.tag === 'title')?.text).toBeTruthy()
      expect(tags.find((tag) => tag.attrs.name === 'description')?.attrs.content).toBeTruthy()
    }
  })
})

describe('renderHeadTags', () => {
  it('emits the marker attribute so client navigation can replace them', () => {
    const html = renderHeadTags(getRouteMeta('/'))
    expect(html).toContain('data-seo')
    expect(html).toContain('<title data-seo="">')
    expect(html).toContain('rel="canonical"')
  })

  it('escapes characters that would break the document', () => {
    const html = renderHeadTags({
      path: '/x',
      title: 'A & B <script>',
      description: 'quote " and ampersand &',
    })
    expect(html).not.toContain('<script>')
    expect(html).toContain('&amp;')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('&quot;')
  })
})

describe('JSON-LD structured data', () => {
  const withJsonLd = {
    path: '/tools/example',
    title: 'Example | ai.hanz.dev',
    description: 'An example page.',
    jsonLd: { '@context': 'https://schema.org', '@type': 'Thing', name: 'Example' },
  }

  it('emits no script tag on a page without structured data', () => {
    const tags = headTags(getRouteMeta('/'))
    expect(tags.find((tag) => tag.tag === 'script')).toBeUndefined()
    expect(renderHeadTags(getRouteMeta('/'))).not.toContain('application/ld+json')
  })

  it('emits a script tag carrying the payload when the route has one', () => {
    const tags = headTags(withJsonLd)
    const script = tags.find((tag) => tag.tag === 'script')
    expect(script?.attrs.type).toBe('application/ld+json')
    expect(script?.text).toContain('"@type":"Thing"')

    const html = renderHeadTags(withJsonLd)
    expect(html).toContain('type="application/ld+json"')
    expect(html).toContain('data-seo')
  })

  it('keeps the payload parseable as JSON', () => {
    const html = renderHeadTags(withJsonLd)
    const match = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)
    expect(match).not.toBeNull()
    expect(JSON.parse(match?.[1] ?? '')).toEqual(withJsonLd.jsonLd)
  })

  it('escapes a payload that would otherwise close the script element', () => {
    const hostile = {
      '@type': 'Thing',
      name: '</script><img src=x onerror="alert(1)">',
    }
    const html = renderHeadTags({ ...withJsonLd, jsonLd: hostile })

    // The literal closing sequence never reaches the document.
    expect(html).not.toContain('</script><img')
    expect(html).toContain('\\u003c/script')

    // Escaping is transparent: the parsed value is unchanged.
    const match = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)
    expect(JSON.parse(match?.[1] ?? '')).toEqual(hostile)
  })
})

describe('faqJsonLd', () => {
  const items = [
    { question: 'How long does REAP take?', answer: 'It depends on the token count.' },
    { question: 'Can I prune on one GPU?', answer: 'Yes, with the layer-wise observer.' },
  ]

  it('produces a FAQPage with one question per item', () => {
    const node = faqJsonLd(items)
    expect(node['@type']).toBe('FAQPage')
    expect(node['@context']).toBe('https://schema.org')

    const entities = node.mainEntity as Array<Record<string, unknown>>
    expect(entities).toHaveLength(2)
    expect(entities[0].name).toBe('How long does REAP take?')
    expect(entities[0]['@type']).toBe('Question')
  })

  it('gives every question an accepted answer', () => {
    const entities = faqJsonLd(items).mainEntity as Array<Record<string, unknown>>
    for (const entity of entities) {
      const answer = entity.acceptedAnswer as Record<string, unknown>
      expect(answer['@type']).toBe('Answer')
      expect(String(answer.text).trim()).not.toBe('')
    }
  })
})

describe('softwareApplicationJsonLd', () => {
  it('describes the tool with its canonical URL', () => {
    const node = softwareApplicationJsonLd(getRouteMeta('/tools/kv-cache-calculator'))
    expect(node['@type']).toBe('SoftwareApplication')
    expect(node.url).toBe(`${SITE_ORIGIN}/tools/kv-cache-calculator`)
    expect(node.isAccessibleForFree).toBe(true)
    expect(node.applicationCategory).toBe('DeveloperApplication')
  })
})

describe('jsonLdGraph', () => {
  it('combines nodes under one context', () => {
    const meta = getRouteMeta('/tools/kv-cache-calculator')
    const graph = jsonLdGraph(
      softwareApplicationJsonLd(meta),
      faqJsonLd([{ question: 'Q', answer: 'A' }]),
    )

    expect(graph['@context']).toBe('https://schema.org')
    const nodes = graph['@graph'] as Array<Record<string, unknown>>
    expect(nodes).toHaveLength(2)
    // The shared context is declared once, on the graph itself.
    for (const node of nodes) {
      expect(node['@context']).toBeUndefined()
    }
    expect(nodes.map((node) => node['@type'])).toEqual(['SoftwareApplication', 'FAQPage'])
  })

  it('round-trips through a script tag and parses back', () => {
    const meta = getRouteMeta('/tools/kv-cache-calculator')
    const jsonLd = jsonLdGraph(
      softwareApplicationJsonLd(meta),
      faqJsonLd([{ question: 'Q', answer: 'A' }]),
    )
    const html = renderHeadTags({ ...meta, jsonLd })
    const match = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)
    const parsed = JSON.parse(match?.[1] ?? '') as Record<string, unknown>
    const types = (parsed['@graph'] as Array<Record<string, unknown>>).map((n) => n['@type'])
    expect(types).toContain('FAQPage')
    expect(types).toContain('SoftwareApplication')
  })
})
