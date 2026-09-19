import { StrictMode } from 'react'
import { renderToString } from 'react-dom/server'
import { StaticRouter } from 'react-router'

import App from '@/App'
import {
  NOT_FOUND_PATH,
  PRERENDER_ATTR,
  PRERENDER_ROUTES,
  PRERENDER_WILDCARD,
  getNotFoundMeta,
  getRouteMeta,
  renderHeadTags,
} from '@/lib/seo'

export interface RenderedRoute {
  html: string
  head: string
}

/** Render one route to static HTML plus its head tags. */
export function render(url: string): RenderedRoute {
  const html = renderToString(
    <StrictMode>
      <StaticRouter location={url}>
        <App />
      </StaticRouter>
    </StrictMode>,
  )

  return { html, head: renderHeadTags(getRouteMeta(url)) }
}

// Re-exported so the prerender script reads route knowledge from one place.
export {
  NOT_FOUND_PATH,
  PRERENDER_ATTR,
  PRERENDER_ROUTES,
  PRERENDER_WILDCARD,
  getNotFoundMeta,
  getRouteMeta,
  renderHeadTags,
}
