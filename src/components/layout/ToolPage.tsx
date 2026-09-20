import { Link } from 'react-router'

/**
 * The breadcrumb and page heading every tool page opens with.
 *
 * The tools are laid out identically, so a new calculator gets the same
 * structure by composing these rather than by copying another page's header.
 */

/** The trail back to the tool index, then the page's own name. */
export function ToolBreadcrumb({ name }: { name: string }) {
  return (
    <nav className="text-muted-foreground text-xs">
      <Link to="/" className="underline-offset-4 hover:underline">
        Tools
      </Link>
      <span aria-hidden="true"> / </span>
      <span>{name}</span>
    </nav>
  )
}

interface ToolHeaderProps {
  title: string
  description: string
  /** Tailwind max-width class. Pages choose how wide their intro runs. */
  widthClass?: string
}

export function ToolHeader({
  title,
  description,
  widthClass = 'max-w-3xl',
}: ToolHeaderProps) {
  return (
    <header className={`flex ${widthClass} flex-col gap-3`}>
      <h1 className="text-2xl font-medium tracking-tight">{title}</h1>
      <p className="text-muted-foreground">{description}</p>
    </header>
  )
}
