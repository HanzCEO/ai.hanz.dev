import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 */
export const COMPRESSION_FAQ: FaqItem[] = [
  {
    question: 'When does summarising a session save money?',
    answer: 'Once the session passes the size named above, and not before that.',
  },
  {
    question: 'Why does session size decide it at all?',
    answer:
      'Because the summary is capped at a share of the window, so it costs the same at any session size while the session it replaces keeps getting dearer.',
  },
  {
    question: 'Why does the cap matter so much?',
    answer:
      'A wider cap means more summary tokens, and every one of them is new text billed at the full input price.',
  },
  {
    question: 'What does the token mix change?',
    answer:
      'It sets how much of your context is a cache hit, which sets how cheaply the kept session grows.',
  },
  {
    question: 'Why is the break-even cap so low on a coding session?',
    answer:
      'Because a coding session barely misses the cache, so the session you already hold is already the cheap option.',
  },
  {
    question: 'Where do the preset rates come from?',
    answer:
      'They are published list prices for one model from each major provider, and every price field is editable.',
  },
]
