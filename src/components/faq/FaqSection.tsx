import { ChevronRight } from 'lucide-react'

import type { FaqItem } from '@/lib/faq'

interface FaqSectionProps {
  /** Id for the heading, which the section points at with aria-labelledby. */
  id: string
  heading: string
  items: readonly FaqItem[]
}

/**
 * The visible questions and answers.
 *
 * Answers are written to stand alone, because a snippet or an AI overview may
 * lift one without the surrounding page. The same list is declared as FAQPage
 * structured data, so the copy a search engine reads is the copy on screen.
 *
 * The questions are collapsed with the native `details` element rather than a
 * scripted accordion, which keeps every answer in the document as the crawler
 * and the structured data see it. Eight long answers open run to several screens
 * of prose below the result.
 */
export default function FaqSection({ id, heading, items }: FaqSectionProps) {
  return (
    <section aria-labelledby={id} className="flex max-w-3xl flex-col gap-4">
      <h2 id={id} className="text-lg font-medium tracking-tight">
        {heading}
      </h2>
      <div className="border-border flex flex-col border-t">
        {items.map((item) => (
          <details key={item.question} className="group border-border border-b py-3">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm font-medium [&::-webkit-details-marker]:hidden">
              {item.question}
              <ChevronRight
                aria-hidden="true"
                className="text-muted-foreground size-4 shrink-0 transition-transform group-open:rotate-90"
              />
            </summary>
            <p className="text-muted-foreground mt-2 text-sm">{item.answer}</p>
          </details>
        ))}
      </div>
    </section>
  )
}
