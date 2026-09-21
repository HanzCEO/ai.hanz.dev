import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 *
 * Answers are written to stand alone, because a snippet or an AI overview may
 * lift one without the surrounding page.
 */
export const REAP_FAQ: FaqItem[] = [
  {
    question: 'How long would it take to REAP a model?',
    answer:
      'It is set almost entirely by the calibration token count, which is the number of samples times the sequence length. A quick pass of 512 samples at 2048 tokens is minutes on a modern card. The recipe published with the REAP repository, 24576 samples at 16384 tokens, is about 403 million tokens and runs for tens of hours on a single GPU. Enter a model and a card above to get the figure for your case.',
  },
  {
    question: 'Can I prune a mixture of experts model on one GPU?',
    answer:
      'Yes. The layer-wise observer added to REAP keeps only one decoder block resident at a time, so the memory requirement is set by the largest single block rather than by the whole model. That is what makes a trillion parameter model tractable on one card. The question is whether one block, plus its activation buffer, fits in your VRAM. The verdict above answers it for the model and card you selected.',
  },
  {
    question: 'How many calibration samples does REAP need?',
    answer:
      'The REAP paper calibrated on 1024 samples at 2048 tokens for models up to 110 billion parameters. It calibrated on 12228 samples at 16384 tokens above that. The REAP repository uses a mix of 24576 samples at 16384 tokens for the released checkpoints. The vLLM llm-compressor example uses 512 samples at 2048 tokens and still ranks experts well. Fewer samples cost proportionally less time but give a noisier saliency ranking, and a dataset that misses a topic can mark the experts for that topic as unimportant.',
  },
  {
    question: 'Does REAP make inference faster?',
    answer:
      'Not on its own. Pruning removes experts from memory, so the model gets smaller and can fit on less hardware. A token still routes to the same number of experts unless the router top-k is reduced as well, so the arithmetic per token is unchanged. Reducing the top-k is what turns a smaller model into a faster one.',
  },
  {
    question: 'What pruning ratio should I use?',
    answer:
      'Published checkpoints use 25, 30, 40, and 50 percent. How much quality is lost depends on how redundant the original experts are. Qwen3-30B-A3B recovered 99.8 percent of its baseline score at 50 percent pruning, while Moonlight-16B-A3B recovered only 17 percent at the same ratio. Calibrate once and the saliency scores can be reused to produce any ratio.',
  },
  {
    question: 'Does REAP work on dense models?',
    answer:
      'No. REAP removes routed experts, so a model with no expert bank has nothing to prune. The calculator says so directly when you enter a dense model.',
  },
]
