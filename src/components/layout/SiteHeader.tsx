import { Link } from 'react-router'

import { REPO_URL, SITE_NAME } from '@/lib/seo'

export default function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-container flex h-14 items-center justify-between gap-4">
        <Link
          to="/"
          className="text-sm font-medium tracking-tight underline-offset-4 hover:underline"
        >
          {SITE_NAME}
        </Link>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          Source
        </a>
      </div>
    </header>
  )
}
