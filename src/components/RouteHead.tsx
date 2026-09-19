import { useEffect } from 'react'
import { useLocation } from 'react-router'

import { applyHead, getRouteMeta } from '@/lib/seo'

/**
 * Renders nothing. The prerender step writes the head tags into the static
 * HTML, and this keeps them in sync once the router takes over.
 */
export default function RouteHead() {
  const { pathname } = useLocation()

  useEffect(() => {
    applyHead(getRouteMeta(pathname))
  }, [pathname])

  return null
}
