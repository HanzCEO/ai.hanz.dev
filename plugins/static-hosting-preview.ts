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
 * Three differences matter. The first two make the prerendered markup disagree
 * with the route the client renders, which React reports as a hydration error.
 * The third decides which URL actually answers 200, and so whether the
 * canonical URLs the site emits are reachable at all.
 *
 *  1. GitHub Pages serves `dist/<route>/index.html` for a request to
 *     `/<route>`. Vite's preview server instead falls back to `dist/index.html`,
 *     so a direct load of a nested route returns the landing page.
 *  2. GitHub Pages serves `dist/404.html` with a 404 status for a path it
 *     cannot resolve. Vite's preview server returns `dist/index.html` with a
 *     200, so an unknown path returns the landing page.
 *  3. GitHub Pages answers a request for a directory that has no trailing slash
 *     with a 301 to the slashed form, and only the slashed form returns 200.
 *     Vite's preview server answers both forms with a 200. Every URL the site
 *     emits carries the slash, so without this redirect a local check could not
 *     tell a correct canonical from one that points at a redirect.
 *
 * This middleware runs ahead of Vite's own handlers and reproduces all three,
 * so local preview predicts production rather than flattering it.
 */
export function staticHostingPreview(distDir = 'dist'): Plugin {
  return {
    name: 'static-hosting-preview',
    configurePreviewServer(server) {
      const root = resolve(server.config.root, distDir)

      server.middlewares.use((req, res, next) => {
        if (req.method !== 'GET' && req.method !== 'HEAD') return next()
        if (!req.url || !isHtmlRequest(req)) return next()

        // The encoded path is kept for the redirect target, so a percent escape
        // in the request survives the round trip unchanged.
        const rawPath = req.url.split('?')[0].split('#')[0]
        const query = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : ''

        let pathname: string
        try {
          pathname = decodeURIComponent(rawPath)
        } catch {
          return next()
        }

        // Anything that is already a real file is left to the static handler.
        if (isFile(join(root, pathname))) return next()

        // A directory holding an index.html is a prerendered route. Trailing
        // slashes are not significant for the lookup, only for the redirect.
        const withoutTrailingSlash = pathname.replace(/\/+$/, '')
        const indexFile = join(root, withoutTrailingSlash, 'index.html')
        if (isFile(indexFile)) {
          if (pathname !== `${withoutTrailingSlash}/`) {
            // The bare form is not a page GitHub Pages serves, so send the
            // request to the slashed form instead of answering it here.
            res.statusCode = 301
            res.setHeader('Location', `${rawPath.replace(/\/+$/, '')}/${query}`)
            res.end()
            return
          }
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