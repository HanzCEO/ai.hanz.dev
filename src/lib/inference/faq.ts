import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 *
 * The first question is the one a reader arrives with, and it is answered
 * directly rather than deferred to the calculator. The weight format is the
 * second, because a quantized checkpoint changes the answer more than any other
 * single field.
 */
export const INFERENCE_FAQ: FaqItem[] = [
  {
    question: 'How much GPU memory does inference need?',
    answer:
      'The weights, the KV cache, the activation buffer, and about 1.5 GiB for the framework. A 70B model needs about 130 GiB of weights in BF16. The same model needs about 35 GiB in MXFP4.',
  },
  {
    question: 'Does a quantized checkpoint need less hardware?',
    answer:
      'Yes. FP8 halves the weight bytes against BF16, and MXFP4 cuts them to about a quarter. The KV cache and the activation buffer do not shrink with the weights. The total therefore falls by less than the weight figure suggests.',
  },
  {
    question: 'What weight format do the current models publish?',
    answer:
      'DeepSeek V4 and V4.1 store their experts in MXFP4 and the rest in FP8. MiMo V2.6 stores most of its weights in MXFP4. The calculator reads the format from the checkpoint config.',
  },
  {
    question: 'Does the KV cache dtype change the GPU answer?',
    answer:
      'Yes, when the cache is large. A narrower cache dtype holds the same tokens in fewer bytes. The cache is then smaller, so the run needs less VRAM. On a long context the cache can be the term that decides which card fits. The cache dtype never changes the weight size.',
  },
  {
    question: 'Why does the page ask for the KV cache first?',
    answer:
      'The cache is part of the footprint the card has to hold. Step 1 fixes the model, the context length, the sequences, and the cache dtypes. Step 2 then answers the hardware question from those values. The two steps therefore never ask for the same value twice. Change any value in step 1 and the hardware answer follows.',
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
    question: 'Is a narrower weight format faster for inference?',
    answer:
      'Decode is bound by memory bandwidth, because each token reads every active weight once. A narrower format reads fewer bytes for each token, so the rate rises. The bandwidth of the card and the active parameter count still set the ceiling. The calculator above reports the roofline for the format you select.',
  },
]
