import { describe, expect, it } from 'vitest'

import { toolPath, tools } from './registry'

describe('toolPath', () => {
  it('ends with a trailing slash', () => {
    // The bare form of a nested path is a 301 on GitHub Pages, so a link built
    // without the slash would cost a redirect hop on every click.
    expect(toolPath('kv-cache-calculator')).toBe('/tools/kv-cache-calculator/')
    expect(toolPath('x')).toBe('/tools/x/')
  })

  it('appends exactly one slash', () => {
    expect(toolPath('kv-cache-calculator').endsWith('//')).toBe(false)
  })
})

describe('tools', () => {
  it('gives every tool a path that ends with a slash', () => {
    for (const tool of tools) {
      expect(toolPath(tool.slug).endsWith('/')).toBe(true)
    }
  })

  it('keeps every tool path distinct', () => {
    const paths = tools.map((tool) => toolPath(tool.slug))
    expect(new Set(paths).size).toBe(paths.length)
  })
})