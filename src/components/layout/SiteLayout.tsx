import type { ReactNode } from 'react'

import SiteFooter from '@/components/layout/SiteFooter'
import SiteHeader from '@/components/layout/SiteHeader'

export default function SiteLayout({ children }: { children: ReactNode }) {
  return (
    <div className="site-shell">
      <SiteHeader />
      <main className="site-main">
        <div className="site-container">{children}</div>
      </main>
      <SiteFooter />
    </div>
  )
}
