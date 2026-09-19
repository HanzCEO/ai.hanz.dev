/**
 * Turns the client build in dist/ into per-route static HTML.
 *
 * Vite emits a single dist/index.html with an empty #root. This script renders
 * each route from the SSR bundle and writes the result back out, so crawlers
 * get real markup instead of an empty shell.
 *
 * Every document is stamped with the route it was rendered for. The client
 * reads that stamp before hydrating, which keeps a mis-served document from
 * producing a hydration error.
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const distDir = join(projectRoot, 'dist')
const serverEntry = join(projectRoot, 'dist-ssr', 'entry-server.js')
const templatePath = join(distDir, 'index.html')

const HEAD_MARKER = '<!--app-head-->'
const ROOT_MARKER = '<div id="root"></div>'
const ORIGIN = 'https://ai.hanz.dev'

/** Where each route's HTML file lives, relative to dist/. */
function outputPathFor(route) {
  if (route === '/') return join(distDir, 'index.html')
  return join(distDir, route.replace(/^\//, ''), 'index.html')
}

function escapeXml(value) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

async function main() {
  const {
    render,
    PRERENDER_ROUTES,
    PRERENDER_WILDCARD,
    PRERENDER_ATTR,
    NOT_FOUND_PATH,
    getRouteMeta,
    getNotFoundMeta,
  } = await import(pathToFileURL(serverEntry).href)

  const template = await readFile(templatePath, 'utf8')

  if (!template.includes(ROOT_MARKER)) {
    throw new Error(`Template is missing ${ROOT_MARKER}`)
  }
  if (!template.includes(HEAD_MARKER)) {
    throw new Error(`Template is missing ${HEAD_MARKER}`)
  }

  // Drop the placeholder title and marker; both are replaced per route.
  const base = template.replace(/<title>.*?<\/title>\s*/s, '').replace(HEAD_MARKER, '')

  /** Builds a full document for one route, stamped with its marker. */
  function buildDocument(route, marker) {
    const { html, head } = render(route)
    return base
      .replace('</head>', `  ${head}\n  </head>`)
      .replace(
        ROOT_MARKER,
        `<div id="root" ${PRERENDER_ATTR}="${marker}">${html}</div>`,
      )
  }

  const written = []

  for (const route of PRERENDER_ROUTES) {
    const target = outputPathFor(route)
    await mkdir(dirname(target), { recursive: true })
    await writeFile(target, buildDocument(route, route), 'utf8')
    written.push(route)
  }

  // GitHub Pages serves 404.html for any path it cannot resolve, so it has to
  // hold the not found page rather than a copy of the landing page. Otherwise
  // the served markup would disagree with the route the client renders.
  await writeFile(
    join(distDir, '404.html'),
    buildDocument(NOT_FOUND_PATH, PRERENDER_WILDCARD),
    'utf8',
  )

  // Regenerate the sitemap from the same route list the prerender used, so the
  // two cannot drift apart. The not found page is deliberately absent.
  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...PRERENDER_ROUTES.map((route) => {
      const meta = getRouteMeta(route)
      return [
        '  <url>',
        `    <loc>${escapeXml(`${ORIGIN}${route}`)}</loc>`,
        `    <title>${escapeXml(meta.title)}</title>`,
        '    <changefreq>monthly</changefreq>',
        '  </url>',
      ].join('\n')
    }),
    '</urlset>',
    '',
  ].join('\n')
  await writeFile(join(distDir, 'sitemap.xml'), sitemap, 'utf8')

  console.log(`prerendered ${written.length} route(s): ${written.join(', ')}`)
  console.log(`wrote 404.html (${getNotFoundMeta().title}) and sitemap.xml`)
}

main().catch((error) => {
  console.error('prerender failed:', error)
  process.exit(1)
})
