import { formatBytes, formatExact, type ComputeResult } from '@/lib/kvcache'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="text-right tabular-nums">{value}</dd>
    </div>
  )
}

export default function BreakdownPanel({ result }: { result: ComputeResult }) {
  return (
    <Accordion type="multiple" className="w-full">
      <AccordionItem value="components">
        <AccordionTrigger className="text-sm">How this was calculated</AccordionTrigger>
        <AccordionContent>
          <ol className="flex flex-col gap-3 text-sm">
            {result.steps.map((step) => (
              <li key={step.label} className="flex flex-col gap-0.5">
                <span className="font-medium">{step.label}</span>
                <span className="text-muted-foreground">{step.detail}</span>
              </li>
            ))}
          </ol>

          <div className="mt-5 flex flex-col gap-3">
            {result.components.map((component) => (
              <div key={component.id} className="border-border rounded-lg border p-3">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="text-sm font-medium">{component.label}</span>
                  <span className="text-sm tabular-nums">{formatBytes(component.totalBytes).text}</span>
                </div>
                <p className="text-muted-foreground mt-1 font-mono text-xs break-words">
                  {component.formula}
                </p>
                <dl className="text-muted-foreground mt-2 text-xs">
                  {component.bytesPerToken > 0 && (
                    <Row
                      label="per token, all layers"
                      value={`${formatExact(component.bytesPerToken)} bytes`}
                    />
                  )}
                  <Row
                    label="per sequence"
                    value={`${formatExact(component.bytesPerSequence)} bytes`}
                  />
                  <Row
                    label={`across ${result.sequenceCount} sequence${result.sequenceCount === 1 ? '' : 's'}`}
                    value={`${formatExact(component.totalBytes)} bytes`}
                  />
                </dl>
                {component.note && (
                  <p className="text-muted-foreground mt-2 text-xs">{component.note}</p>
                )}
              </div>
            ))}
          </div>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="config">
        <AccordionTrigger className="text-sm">Config values used</AccordionTrigger>
        <AccordionContent>
          <dl className="text-sm">
            {result.constants.map((constant) => (
              <div
                key={`${constant.key}-${String(constant.value)}`}
                className="border-border flex flex-col gap-0.5 border-b py-2 last:border-b-0"
              >
                <div className="flex items-baseline justify-between gap-4">
                  <dt className="font-mono text-xs">{constant.key}</dt>
                  <dd className="tabular-nums">{String(constant.value)}</dd>
                </div>
                <p className="text-muted-foreground text-xs">{constant.source}</p>
              </div>
            ))}
          </dl>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="assumptions">
        <AccordionTrigger className="text-sm">
          Assumptions and caveats ({result.assumptions.length})
        </AccordionTrigger>
        <AccordionContent>
          <ul className="text-muted-foreground flex list-disc flex-col gap-2 pl-4 text-sm">
            {result.assumptions.map((assumption) => (
              <li key={assumption}>{assumption}</li>
            ))}
          </ul>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
