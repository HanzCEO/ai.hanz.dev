import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 *
 * The first question is the one a reader arrives with, and it is answered
 * directly rather than deferred to the calculator.
 */
export const INFERENCE_FAQ: FaqItem[] = [
  {
    question: 'How much GPU memory does FP16 or BF16 inference need?',
    answer:
      'Two bytes for each parameter, plus the KV cache, plus about 2 GiB for the framework. A 70B model therefore needs about 141 GiB for the weights alone. The calculator above adds the cache for your context length and your sequence count. It therefore reports the figure for your case and not for the weights alone.',
  },
  {
    question: 'Why does the page ask for the KV cache first?',
    answer:
      'The cache is part of the footprint the card has to hold. Step 1 fixes the model, the context length, the sequences, and the cache dtypes. Step 2 then answers the hardware question from those values. The two steps therefore never ask for the same value twice. Change any value in step 1 and the hardware answer follows.',
  },
  {
    question: 'Does the KV cache dtype change the GPU answer?',
    answer:
      'Yes, when the cache is large. A narrower cache dtype holds the same tokens in fewer bytes. The cache is then smaller, so the run needs less VRAM. On a long context the cache can be the term that decides which card fits. The cache dtype never changes the weight size, because the weights are still served in FP16 or BF16.',
  },
  {
    question: 'Do FP16 and BF16 need different GPU hardware?',
    answer:
      'No. FP16 and BF16 are both two bytes for each weight. A model served in either one therefore needs the same VRAM and the same bandwidth. The two differ in numeric range and not in size. BF16 has a wider exponent, so it holds large activations without overflowing. It is the safer choice for a large model.',
  },
  {
    question: 'What GPU do I need for a 70B model in BF16?',
    answer:
      'The weights alone take about 141 GiB, so one 192 GB card holds them. Several smaller cards can share them instead. A card of 80 GB needs 2 or more cards. The KV cache and the framework reserve add to that figure. The calculator above reports the exact configuration for your context length.',
  },
  {
    question: 'When does inference need several GPUs?',
    answer:
      'Inference needs several GPUs when the weights and the cache do not fit one card. Tensor parallelism divides the weights and the cache across the cards. The activation buffer and the framework reserve stay in full on every card. A second card therefore does not halve the footprint. Tensor parallelism across more than one node also needs a fast interconnect, because the cards exchange activations at every layer.',
  },
  {
    question: 'How many concurrent requests can one GPU serve?',
    answer:
      'The KV cache decides it. Every concurrent sequence holds its own cache. The count therefore follows from the VRAM left over once the weights are resident. Doubling the context length halves the sequence count that fits. The calculator above reports the sequence count the free VRAM allows.',
  },
  {
    question: 'Is FP16 or BF16 faster for inference?',
    answer:
      'They run at the same speed. Decode is bound by memory bandwidth, because each token reads every active weight once. Both precisions read two bytes for each weight. The bandwidth of the card and the number of active parameters set the speed. The choice between these two precisions does not change it.',
  },
]
