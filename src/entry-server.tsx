import { StrictMode } from 'react'
import { renderToString } from 'react-dom/server'
import { StaticRouter } from 'react-router'

import App from '@/App'
import { PRERENDER_ROUTES, getRouteMeta, renderHeadTags } from '@/lib/seo'

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

export { PRERENDER_ROUTES, getRouteMeta, renderHeadTags }
