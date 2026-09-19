import ToolCard from '@/components/ToolCard'
import { tools } from '@/tools/registry'

export default function Home() {
  return (
    <div className="flex flex-col gap-10">
      <section className="flex max-w-2xl flex-col gap-3">
        <h1 className="text-2xl font-medium tracking-tight">Tools for AI developers</h1>
      </section>

      <section aria-labelledby="tool-list-heading" className="flex flex-col gap-4">
        <h2 id="tool-list-heading" className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          {tools.length === 1 ? '1 tool' : `${tools.length} tools`}
        </h2>
        <ul className="grid list-none gap-4 p-0 sm:grid-cols-2">
          {tools.map((tool) => (
            <li key={tool.slug} className="relative">
              <ToolCard tool={tool} />
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
