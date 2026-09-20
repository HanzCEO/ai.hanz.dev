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
 */
export default function FaqSection({ id, heading, items }: FaqSectionProps) {
  return (
    <section aria-labelledby={id} className="flex max-w-3xl flex-col gap-5">
      <h2 id={id} className="text-lg font-medium tracking-tight">
        {heading}
      </h2>
      <div className="flex flex-col gap-5">
        {items.map((item) => (
          <div key={item.question}>
            <h3 className="text-sm font-medium">{item.question}</h3>
            <p className="text-muted-foreground mt-1 text-sm">{item.answer}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
