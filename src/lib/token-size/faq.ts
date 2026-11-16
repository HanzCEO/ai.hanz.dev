import type { FaqItem } from '@/lib/faq'

/**
 * The questions the page answers in prose, and the same questions it declares
 * as FAQPage structured data. One list feeds both, so the visible copy and the
 * markup a search engine reads can never disagree.
 */
export const TOKEN_SIZE_FAQ: FaqItem[] = [
  {
    question: 'What does this calculator answer?',
    answer:
      'It turns a token count into a byte count for one specific tokenizer. You pick a model, type its org/name, choose how many tokens you are interested in, and choose whether those tokens are stored as text, JSONL, Parquet, or Arrow. The page then reports that many tokens as bytes, KB, MB, GB, TB, and PB — the number a dataset download, a stored prompt buffer, or a pricing bill is actually measured in.',
  },
  {
    question: 'Why does the answer depend on the model?',
    answer:
      'Tokens are not bytes. Each tokenizer splits text differently, so the same 1,000,000 tokens cover different amounts of text depending on the vocabulary. A byte-level BPE stores spaces as a single character while a SentencePiece tokenizer marks them with ▁, vocabularies differ in size, and the merge rankings change how greedily words split. The page reads the model\u2019s own tokenizer.json from the hub so the figure follows that model rather than a global rule of thumb.',
  },
  {
    question: 'How is the bytes-per-token figure measured?',
    answer:
      'For the two BPE families, the tokenizer is actually run over a fixed sample bundled into the page: English prose, function definitions, numbers, symbols, URLs, and a paragraph of CJK text. The corpus byte count is divided by the number of tokens the model\u2019s own pretokenizer and merge ranks produced. The result is exact for that corpus, and the same corpus is used for every model, so comparisons across models are fair. Real text drifts from the figure because a page of code or a chat log will not match the sample mix.',
  },
  {
    question: 'How big is the sample text the measurement uses?',
    answer:
      'Roughly three thousand bytes. That is large enough for the merge algorithm to behave the way it does on real text while staying small enough that the analysis runs in the browser in a few milliseconds for any model you type.',
  },
  {
    question: 'Is the answer different for unigram and WordPiece tokenizers?',
    answer:
      'Yes, because those families cannot be replayed the way BPE can. A unigram tokenizer has no merge ranks, so its bytes-per-token figure is the expected byte length under the tokenizer\u2019s own score distribution — the mean it was trained toward. A WordPiece or unrecognised tokenizer has neither merges nor scores, so its figure is the mean decoded piece length in its vocabulary, and the page marks it as an estimate.',
  },
  {
    question: 'Are these GB or GiB?',
    answer:
      'The table uses decimal units — one GB is 1,000,000,000 bytes — which is how storage and API billing usually quote sizes. The headline also prints the same byte count in binary units, which is how memory is usually quoted, alongside the exact byte count, so neither reading is lost.',
  },
  {
    question: 'Why does the file format change the answer so much?',
    answer:
      'Text and token ids are different things. The text reading counts the UTF-8 bytes the tokenizer was measured on, which is what a download of raw text costs. A tokenized file stores token ids instead, so its size follows the vocabulary width rather than the language: a vocabulary of 65,536 ids or fewer fits each token in two bytes, and a larger one needs four. Arrow keeps those bytes uncompressed, Parquet compresses the column and usually lands near 60% of the raw width, and JSONL writes each id as decimal text with a comma between, so its size depends on how many digits the ids take.',
  },
  {
    question: 'Which storage format should I pick?',
    answer:
      'Pick Text when you are sizing the corpus the tokens were produced from, for example a dataset you would download before tokenizing. Pick Parquet or Arrow when you keep a pre-tokenized dataset, since streaming token ids off disk is how training pipelines avoid the tokenizer at step time. Parquet is the usual choice because it is compact and readable from most tools; Arrow is uncompressed and memory-mappable, which is why it is quoted as the exact width.',
  },
  {
    question: 'Why does some models\u2019 tab say it needs a token?',
    answer:
      'Several publishers gate their tokenizer files. If a repo returns a gated error, paste a hub token into the token field. The token is sent only to the hub you selected, held in memory, and never written to the page URL.',
  },
  {
    question: 'Why is there no tokenizer for my model?',
    answer:
      'Not every repo publishes a tokenizer.json — encoder-only classifiers, vision models, and some hosted checkpoints ship only weights or no text pipeline at all. If the hub reports that no tokenizer.json exists, pick the base text model of the same family, which almost always shares the same tokenizer.',
  },
]