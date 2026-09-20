/**
 * The shape of a question and answer pair.
 *
 * A tool's FAQ is rendered on the page and also declared as FAQPage structured
 * data, from one list. The type lives here rather than with the SEO module so
 * that a shared FAQ component and a tool's own FAQ data do not have to depend
 * on how the page is indexed.
 */
export interface FaqItem {
  question: string
  answer: string
}
