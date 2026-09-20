import type { ReactNode } from 'react'

import { formatExact } from '@/lib/format'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'

/**
 * The "show your working" panel that sits under every result.
 *
 * Each calculator explains itself the same way: the steps that produced the
 * number, the values read from the model config, and the assumptions that would
 * change the answer. Only the extra sections differ, so those are passed in as
 * slots rather than duplicated.
 */

export interface BreakdownStep {
  label: string
  detail: string
}

export interface BreakdownConstant {
  key: string
  value: number | string
  source: string
}

/** One label and value on a line, for a sub-table inside a panel. */
export function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="text-right tabular-nums">{value}</dd>
    </div>
  )
}

/**
 * An extra panel a caller wants alongside the three shared ones.
 *
 * Rendered between the steps and the constants, which is where a tool's own
 * detail belongs: after the summary of how the number was reached and before
 * the raw inputs it was reached from.
 */
export function BreakdownPanel({
  value,
  label,
  children,
}: {
  value: string
  label: string
  children: ReactNode
}) {
  return (
    <AccordionItem value={value}>
      <AccordionTrigger className="text-sm">{label}</AccordionTrigger>
      <AccordionContent>{children}</AccordionContent>
    </AccordionItem>
  )
}

interface BreakdownProps {
  steps: BreakdownStep[]
  constants: BreakdownConstant[]
  assumptions: string[]
  /** Trigger label for the constants panel, which each tool words its own way. */
  constantsLabel?: string
  /** Rendered inside the steps panel, below the list. */
  stepsExtra?: ReactNode
  /** Rendered as a panel of its own, between the steps and the constants. */
  extraPanels?: ReactNode
}

export default function Breakdown({
  steps,
  constants,
  assumptions,
  constantsLabel = 'Config values used',
  stepsExtra,
  extraPanels,
}: BreakdownProps) {
  return (
    <Accordion type="multiple" className="w-full">
      <AccordionItem value="steps">
        <AccordionTrigger className="text-sm">How this was calculated</AccordionTrigger>
        <AccordionContent>
          <ol className="flex flex-col gap-3 text-sm">
            {steps.map((step) => (
              <li key={step.label} className="flex flex-col gap-0.5">
                <span className="font-medium">{step.label}</span>
                <span className="text-muted-foreground">{step.detail}</span>
              </li>
            ))}
          </ol>
          {stepsExtra}
        </AccordionContent>
      </AccordionItem>

      {extraPanels}

      <AccordionItem value="config">
        <AccordionTrigger className="text-sm">{constantsLabel}</AccordionTrigger>
        <AccordionContent>
          <dl className="text-sm">
            {constants.map((constant) => (
              <div
                key={`${constant.key}-${String(constant.value)}`}
                className="border-border flex flex-col gap-0.5 border-b py-2 last:border-b-0"
              >
                <div className="flex items-baseline justify-between gap-4">
                  <dt className="font-mono text-xs">{constant.key}</dt>
                  <dd className="tabular-nums">
                    {typeof constant.value === 'number'
                      ? formatExact(constant.value)
                      : String(constant.value)}
                  </dd>
                </div>
                <p className="text-muted-foreground text-xs">{constant.source}</p>
              </div>
            ))}
          </dl>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="assumptions">
        <AccordionTrigger className="text-sm">
          Assumptions and caveats ({assumptions.length})
        </AccordionTrigger>
        <AccordionContent>
          <ul className="text-muted-foreground flex list-disc flex-col gap-2 pl-4 text-sm">
            {assumptions.map((assumption) => (
              <li key={assumption}>{assumption}</li>
            ))}
          </ul>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
