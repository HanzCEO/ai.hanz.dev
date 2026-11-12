import type { ModelPreset } from '@/components/model-source/ModelSourceFields'

/**
 * Quick picks for the model id field.
 *
 * The list is chosen to cover the tokenizer families the page reports, not the
 * strongest models. A reader can see a byte-level BPE beside a SentencePiece
 * BPE, a unigram model, and a WordPiece model, and watch the same token count
 * become a different size. Every entry points at a repo whose tokenizer.json is
 * public, apart from the gated one, which says so in its note.
 */
export const TOKENIZER_PRESETS: ModelPreset[] = [
  { id: 'Qwen/Qwen3-8B', note: 'Byte-level BPE, 151k vocab' },
  { id: 'deepseek-ai/DeepSeek-V3', note: 'Byte-level BPE, 129k vocab' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', note: 'Byte-level BPE, 128k vocab. Gated, needs a token.' },
  { id: 'openai-community/gpt2', note: 'Byte-level BPE, 50k vocab, the classic GPT-2 split' },
  { id: 'microsoft/Phi-3-mini-4k-instruct', note: 'Byte-level BPE, 32k vocab' },
  { id: 'mistralai/Mistral-7B-v0.1', note: 'SentencePiece BPE, 32k vocab' },
  { id: 'NousResearch/Llama-2-7b-hf', note: 'SentencePiece BPE, 32k vocab, a mirror of Llama 2' },
  { id: 'unsloth/gemma-2-2b-it', note: 'SentencePiece BPE, 256k vocab' },
  { id: 't5-base', note: 'Unigram, 32k vocab, scored rather than merged' },
  { id: 'bert-base-uncased', note: 'WordPiece, 31k vocab, vocabulary average only' },
]