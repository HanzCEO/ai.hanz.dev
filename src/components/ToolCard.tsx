import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { toolPath, type Tool } from '@/tools/registry'

export default function ToolCard({ tool }: { tool: Tool }) {
  const Icon = tool.icon

  return (
    <Card className="group h-full gap-4 transition-colors hover:border-foreground/30">
      <CardHeader>
        <div className="mb-2 flex items-center justify-between gap-2">
          <Icon className="size-4 text-muted-foreground" aria-hidden="true" />
          {tool.status === 'planned' && <Badge variant="secondary">Planned</Badge>}
        </div>
        <CardTitle className="text-base">
          <Link to={toolPath(tool.slug)} className="after:absolute after:inset-0">
            {tool.name}
          </Link>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">{tool.description}</p>
        <span className="inline-flex items-center gap-1 text-sm font-medium">
          Open
          <ArrowRight
            className="size-3.5 transition-transform group-hover:translate-x-0.5"
            aria-hidden="true"
          />
        </span>
      </CardContent>
    </Card>
  )
}
