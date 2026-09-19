import { REPO_URL } from '@/lib/seo'

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-container flex flex-wrap items-center justify-between gap-2 py-6 text-xs text-muted-foreground">
        <p>Every tool runs in your browser. Nothing is sent to a server of ours.</p>
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="hover:text-foreground">
          Source
        </a>
      </div>
    </footer>
  )
}
