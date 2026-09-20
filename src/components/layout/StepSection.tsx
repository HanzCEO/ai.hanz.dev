import { ChevronDown } from 'lucide-react'
import type { ReactNode } from 'react'

import { Card, CardContent } from '@/components/ui/card'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'

/**
 * One numbered step of a calculator that answers a question in stages.
 *
 * A page that needs several inputs before it can answer shows them as steps.
 * Each step is a section that can be opened and closed, and a closed step shows
 * a summary of what it decided rather than nothing. The content is unmounted
 * while closed, so a long form costs nothing until a reader opens it.
 */

interface StepSectionProps {
  /** The step number, shown beside the title. */
  step: number
  title: string
  description: string
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Blocks the step until an earlier one has produced something to use. */
  disabled?: boolean
  /** Shown in place of the content while the step is closed. */
  collapsedSummary?: ReactNode
  children: ReactNode
}

export default function StepSection({
  step,
  title,
  description,
  open,
  onOpenChange,
  disabled = false,
  collapsedSummary,
  children,
}: StepSectionProps) {
  const triggerId = `step-${step}-trigger`
  const panelId = `step-${step}-panel`

  return (
    <Collapsible open={open && !disabled} onOpenChange={onOpenChange} asChild>
      <section className="flex flex-col gap-3">
        <h2 className="flex flex-col gap-1">
          <CollapsibleTrigger
            id={triggerId}
            disabled={disabled}
            aria-controls={panelId}
            className="group flex items-start gap-3 text-left disabled:cursor-not-allowed disabled:opacity-60"
          >
            <span
              className="border-border text-muted-foreground mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full border text-xs tabular-nums"
              aria-hidden="true"
            >
              {step}
            </span>
            <span className="flex flex-1 flex-col gap-1">
              <span className="text-lg font-medium tracking-tight">{title}</span>
              <span className="text-muted-foreground text-sm">{description}</span>
            </span>
            <ChevronDown
              className="text-muted-foreground mt-1 size-4 shrink-0 transition-transform group-data-[state=open]:rotate-180"
              aria-hidden="true"
            />
          </CollapsibleTrigger>
        </h2>

        <CollapsibleContent id={panelId} aria-labelledby={triggerId}>
          {children}
        </CollapsibleContent>

        {!open && collapsedSummary && (
          <Card>
            <CardContent className="text-muted-foreground text-sm">{collapsedSummary}</CardContent>
          </Card>
        )}
      </section>
    </Collapsible>
  )
}
