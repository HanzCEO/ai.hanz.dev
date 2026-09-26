import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 */
export const COMPRESSION_FAQ: FaqItem[] = [
  {
    question: 'When does summarising a session save money?',
    answer:
      'When the summary keeps less of the context than the break-even share above.',
  },
  {
    question: 'Why is the break-even share so low?',
    answer:
      'Because a cache hit is cheap, so the session you already hold is already the cheap option.',
  },
  {
    question: 'What does the token mix change?',
    answer:
      'It sets how much of your context is a cache hit, which is what makes a long session cheap.',
  },
  {
    question: 'Why do the two cost lines only meet at zero?',
    answer:
      'Both grow in step with the context size, so nothing changes between them except which one is steeper.',
  },
  {
    question: 'Where do the preset rates come from?',
    answer:
      'They are published list prices for one model from each major provider, and every price field is editable.',
  },
]
