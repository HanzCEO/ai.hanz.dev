/**
 * Turns the client build in dist/ into per-route static HTML.
 *
 * Vite emits a single dist/index.html with an empty #root. This script renders
 * each route from the SSR bundle and writes the result back out, so crawlers
 * get real markup instead of an empty shell.
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

/** Where each route's HTML file lives, relative to dist/. */
function outputPathFor(route) {
  if (route === '/') return join(distDir, 'index.html')
  return join(distDir, route.replace(/^\//, ''), 'index.html')
}

function escapeXml(value) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

async function main() {
  const { render, PRERENDER_ROUTES, getRouteMeta } = await import(pathToFileURL(serverEntry).href)

  const template = await readFile(templatePath, 'utf8')

  if (!template.includes(ROOT_MARKER)) {
    throw new Error(`Template is missing ${ROOT_MARKER}`)
  }
  if (!template.includes(HEAD_MARKER)) {
    throw new Error(`Template is missing ${HEAD_MARKER}`)
  }

  // Drop the placeholder title and marker; both are replaced per route.
  const base = template
    .replace(/<title>.*?<\/title>\s*/s, '')
    .replace(HEAD_MARKER, '')

  const written = []

  for (const route of PRERENDER_ROUTES) {
    const { html, head } = render(route)

    const page = base
      .replace('</head>', `  ${head}\n  </head>`)
      .replace(ROOT_MARKER, `<div id="root">${html}</div>`)

    const target = outputPathFor(route)
    await mkdir(dirname(target), { recursive: true })
    await writeFile(target, page, 'utf8')
    written.push(route)
  }

  // GitHub Pages serves 404.html for unknown paths. Point it at the landing
  // page so a deep link into a missing route still renders the app shell.
  const landing = await readFile(join(distDir, 'index.html'), 'utf8')
  await writeFile(join(distDir, '404.html'), landing, 'utf8')

  // Regenerate the sitemap from the same route list the prerender used, so the
  // two cannot drift apart.
  const origin = 'https://ai.hanz.dev'
  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...PRERENDER_ROUTES.map((route) => {
      const meta = getRouteMeta(route)
      return [
        '  <url>',
        `    <loc>${escapeXml(`${origin}${route}`)}</loc>`,
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
  console.log('wrote 404.html and sitemap.xml')
}

main().catch((error) => {
  console.error('prerender failed:', error)
  process.exit(1)
})
