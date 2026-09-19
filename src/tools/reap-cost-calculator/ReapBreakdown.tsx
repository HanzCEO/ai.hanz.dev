import { formatBytes, formatExact } from '@/lib/kvcache'
import type { ReapResult } from '@/lib/reap'

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

export default function ReapBreakdown({ result }: { result: ReapResult }) {
  if (!result.moe || !result.shape) return null

  return (
    <Accordion type="multiple" className="w-full">
      <AccordionItem value="steps">
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
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="memory">
        <AccordionTrigger className="text-sm">Memory in detail</AccordionTrigger>
        <AccordionContent>
          <dl className="text-sm">
            <Row label="One expert block" value={formatBytes(result.perMoELayerBytes).text} />
            <Row label="Activation buffer" value={formatBytes(result.activationBytes).text} />
            <Row
              label="Framework and kernels"
              value={formatBytes(result.peakVramBytes - result.perMoELayerBytes - result.activationBytes).text}
            />
            <Row label="Peak resident" value={formatBytes(result.peakVramBytes).text} />
            <Row label="Card memory" value={formatBytes(result.vramBytes).text} />
            <Row label="All weights, on storage" value={formatBytes(result.weightBytes).text} />
          </dl>
          <p className="text-muted-foreground mt-3 text-xs">
            The layer-wise observer holds one decoder block at a time, so the peak is set by the
            block and its activations, not by the whole model. The full weight set still has to live
            somewhere, which is what the storage row is for.
          </p>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="config">
        <AccordionTrigger className="text-sm">Values read from the config</AccordionTrigger>
        <AccordionContent>
          <dl className="text-sm">
            {result.constants.map((constant) => (
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
