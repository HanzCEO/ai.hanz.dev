import { Link } from 'react-router'

/**
 * Tool page shell. The calculator itself lands in the slot below; the copy
 * around it is real so the prerendered HTML carries something worth indexing.
 */
export default function KvCacheCalculator() {
  return (
    <div className="flex flex-col gap-8">
      <nav className="text-xs text-muted-foreground">
        <Link to="/" className="underline-offset-4 hover:underline">
          Tools
        </Link>
        <span aria-hidden="true"> / </span>
        <span>KV Cache Calculator</span>
      </nav>

      <header className="flex max-w-2xl flex-col gap-3">
        <h1 className="text-2xl font-medium tracking-tight">KV Cache Calculator</h1>
        <p className="text-muted-foreground">
          Work out how much memory the KV cache takes for a given model, context length, and
          number of sequences. The model config is fetched from HuggingFace or ModelScope, so the
          answer follows the real architecture instead of a hardcoded table.
        </p>
      </header>

      {/* calculator */}
    </div>
  )
}
