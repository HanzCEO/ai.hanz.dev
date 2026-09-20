import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 *
 * Answers are written to stand alone, because a snippet or an AI overview may
 * lift one without the surrounding page.
 */
export const DSPARK_FAQ: FaqItem[] = [
  {
    question: 'What does it cost to train a DSpark drafter?',
    answer:
      'It depends almost entirely on two choices: how many training tokens you use, and whether you precompute the target cache or capture it online. Offline training over the published recipe writes tens of terabytes of target hidden states but trains on one consumer card. Online capture writes nothing and needs the whole target model resident for the entire run. Enter a target above and the calculator answers both.',
  },
  {
    question: 'Why is the target cache so large?',
    answer:
      'The drafter is supervised on the target model itself, not just its output tokens, so training needs the target hidden states for every token in the training set. Each token stores one bf16 vector per captured layer, plus the last hidden state, plus its token id and two masks. Five captured layers on a 2560 wide model is about 30 kilobytes per token, which over a billion tokens is tens of terabytes.',
  },
  {
    question: 'How do I make the cache smaller?',
    answer:
      'Capture fewer target layers. The cache is proportional to the captured layer count, so going from five layers to two cuts it by roughly 60 percent. The paper found that a shallower drafter still beats a fully parallel one, so the layer count is a real tuning knob rather than a fixed requirement. A smaller training set is the other lever, at the cost of a weaker drafter.',
  },
  {
    question: 'Can I train DSpark on one GPU?',
    answer:
      'Yes, in offline mode, and that is the usual way. The drafter itself is small: five blocks of backbone plus a projection and two heads. The frozen target is the large part, and offline training runs the target once during cache preparation and then lets it go. Online capture needs the target resident for the whole run, which on a large model is the difference between one card and several.',
  },
  {
    question: 'How long does DSpark training take?',
    answer:
      'On the published recipe, with the cache on fast host memory, the arithmetic and the cache reads come out close to the same size, so the run takes a similar number of hours either way. Move the cache to a network share and the run becomes entirely I/O bound and takes days. The bound line above says which of the two your settings produce.',
  },
  {
    question: 'Do I need to regenerate the training data?',
    answer:
      'Yes, and it is not optional. Training is teacher forced, so the drafter learns to imitate the target distribution on the target own outputs. If you train against a dataset written by a different model, or by a different mode of the same model, the acceptance length at inference drops. Regenerate the answers with the exact target checkpoint you intend to serve.',
  },
  {
    question: 'Which target models can a DSpark drafter attach to?',
    answer:
      'Any decoder whose config exposes a hidden size, a depth, and a usable embedding and language model head, which in practice means any modern decoder. The published drafters cover Qwen3 and Gemma. A drafter is bound to the exact target weights it was trained against, so a fine tune, a merge, or a quantized variant needs its own drafter.',
  },
  {
    question: 'Does this estimate the serving speedup?',
    answer:
      'No. This calculator covers training only. The speedup you get at serving time is set by the acceptance length the drafter achieves, which depends on the data you trained on and the verification schedule you deploy. The paper reports 60 to 85 percent faster per user generation at matched throughput against its production baseline.',
  },
]
