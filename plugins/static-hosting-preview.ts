import { readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

import type { Plugin } from 'vite'

function isFile(path: string): boolean {
  try {
    return statSync(path).isFile()
  } catch {
    return false
  }
}

function isHtmlRequest(req: { headers: Record<string, unknown> }): boolean {
  const accept = req.headers.accept
  if (typeof accept !== 'string' || accept === '') return true
  return accept.includes('text/html') || accept.includes('*/*')
}

/**
 * Makes `vite preview` serve the build the way GitHub Pages does.
 *
 * Two differences matter, and both make the prerendered markup disagree with
 * the route the client renders, which React reports as a hydration error:
 *
 *  1. GitHub Pages serves `dist/<route>/index.html` for a request to
 *     `/<route>`. Vite's preview server instead falls back to `dist/index.html`,
 *     so a direct load of a nested route returns the landing page.
 *  2. GitHub Pages serves `dist/404.html` with a 404 status for a path it
 *     cannot resolve. Vite's preview server returns `dist/index.html` with a
 *     200, so an unknown path returns the landing page.
 *
 * This middleware runs ahead of Vite's own handlers and reproduces both, so
 * local preview predicts production rather than flattering it.
 */
export function staticHostingPreview(distDir = 'dist'): Plugin {
  return {
    name: 'static-hosting-preview',
    configurePreviewServer(server) {
      const root = resolve(server.config.root, distDir)

      server.middlewares.use((req, res, next) => {
        if (req.method !== 'GET' && req.method !== 'HEAD') return next()
        if (!req.url || !isHtmlRequest(req)) return next()

        let pathname: string
        try {
          pathname = decodeURIComponent(req.url.split('?')[0].split('#')[0])
        } catch {
          return next()
        }

        // Anything that is already a real file is left to the static handler.
        if (isFile(join(root, pathname))) return next()

        // A directory holding an index.html is a prerendered route. Trailing
        // slashes are not significant, so /a and /a/ resolve the same way.
        const withoutTrailingSlash = pathname.replace(/\/+$/, '')
        const indexFile = join(root, withoutTrailingSlash, 'index.html')
        if (isFile(indexFile)) {
          const query = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : ''
          req.url = `${withoutTrailingSlash}/index.html${query}`
          return next()
        }

        // Everything else is a miss, which GitHub Pages answers with 404.html.
        const notFound = join(root, '404.html')
        if (isFile(notFound)) {
          const body = readFileSync(notFound)
          res.statusCode = 404
          res.setHeader('Content-Type', 'text/html; charset=utf-8')
          res.setHeader('Content-Length', String(body.byteLength))
          res.end(req.method === 'HEAD' ? undefined : body)
          return
        }

        next()
      })
    },
  }
}
