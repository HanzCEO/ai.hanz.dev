import { StrictMode } from 'react'
import { createRoot, hydrateRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'

import '@/index.css'
import '@/styles/main.scss'

import App from '@/App'
import { PRERENDER_ATTR, prerenderMarkerFor } from '@/lib/seo'

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Missing #root element')
}

const tree = (
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
)

/**
 * Only hydrate markup that belongs to the route being rendered.
 *
 * The prerender step stamps each document with the route it rendered. If the
 * server hands back a different document, for example a host that falls back to
 * index.html or 404.html for a path it cannot resolve, the markup would
 * disagree with the client and React would report a hydration error. Comparing
 * the stamp first turns that into a clean client render instead.
 */
const prerenderedRoute = rootElement.getAttribute(PRERENDER_ATTR)
const expectedRoute = prerenderMarkerFor(window.location.pathname)

const canHydrate =
  rootElement.hasChildNodes() && prerenderedRoute !== null && prerenderedRoute === expectedRoute

if (canHydrate) {
  hydrateRoot(rootElement, tree)
} else {
  // Discard markup that belongs to another route before mounting.
  rootElement.replaceChildren()
  createRoot(rootElement).render(tree)
}
