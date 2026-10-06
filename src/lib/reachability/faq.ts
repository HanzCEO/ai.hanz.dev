import type { FaqItem } from '../faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 */
export const REACHABILITY_FAQ: FaqItem[] = [
  {
    question: 'What does reachability mean on this page?',
    answer:
      'It is the smallest card in the hardware directory on this site that can hold the whole run, the number of cards here that can hold it, and the decode rate that configuration reaches. The run is the weights, the KV cache at the chosen context length and sequence count, an activation buffer, and a framework reserve.',
  },
  {
    question: 'Where do the models and the scores come from?',
    answer:
      'The model list, the Decision Index, and the measured latency come from the clef-evals Decision Model Leaderboard. The board is scored on one RTX PRO 6000, so its latency figure is not the decode rate shown here.',
  },
  {
    question: 'Why is the decode rate different from the measured latency?',
    answer:
      'The latency on the board is one end to end call, which includes prefill, the decision head, and the client round trip. The rate here is a bandwidth roofline for decode alone, which is the best a card can do once the prompt is read.',
  },
  {
    question: 'Why can a model not be sized at all?',
    answer:
      'Either the checkpoint publishes no config this site can read, or the config describes an architecture that is not a decoder transformer. A closed hosted API publishes no checkpoint. An encoder only classifier publishes one, but it has no KV cache and no decode loop, so the sizing model does not apply to it.',
  },
  {
    question: 'How accurate is the weight footprint?',
    answer:
      'It is config arithmetic rather than a measurement. The calculator counts the attention blocks, the dense feed forward, the expert bank, the router, and the embedding tables from the published config. It does not count a multimodal vision tower, a multi token prediction block, or the savings of layers that share weights, so a model with any of those can be a few percent off in either direction. The breakdown shows the parameter count the config produced beside the count the leaderboard reports.',
  },
  {
    question: 'Why does forcing a weight format change which cards fit?',
    answer:
      'The bytes for each weight follow the format. A checkpoint published in BF16 costs two bytes for each weight, while an FP8 or a four bit copy costs one byte or half a byte. Forcing a narrower format prices the same parameter count in fewer bytes, so smaller cards enter the fitting set.',
  },
  {
    question: 'Why does the context length change the answer?',
    answer:
      'The KV cache grows with the context length and the sequence count, and it has to be resident beside the weights. A model that fits on a small card at a short context can need a larger card at a long one. The activation buffer follows the sequence count instead, because it holds one token of intermediates for each sequence in flight.',
  },
  {
    question: 'What does the card count mean when one card is not enough?',
    answer:
      'The weights and the KV cache divide across the tensor parallel ranks, while the activation buffer and the framework reserve stay in full on every card. Two cards therefore do not halve the footprint. The page reports the fewest cards that hold the run, and then the smallest card that reaches that count.',
  },
]
