import { describe, expect, it } from 'vitest'

import { analyzeTokenizer } from '@/lib/token-size/tokenizer'
import { TOKENIZER_CORPUS } from '@/lib/token-size/corpus'

// Reference counts come from the `tokenizers` Rust library on the same corpus.
// They pin the shipped analyzer against the real tokenizers without adding a
// native dependency to this repo.
const REFERENCES: Array<{ repo: string; tokens: number; family: string }> = [
  { repo: 'openai-community/gpt2', tokens: 921, family: 'byte_level_bpe' },
  { repo: 'Qwen/Qwen3-8B', tokens: 647, family: 'byte_level_bpe' },
  { repo: 'unsloth/Llama-3.2-1B-Instruct', tokens: 642, family: 'byte_level_bpe' },
  { repo: 'deepseek-ai/DeepSeek-Coder-6.7B-base', tokens: 720, family: 'byte_level_bpe' },
  { repo: 'mistralai/Mistral-7B-v0.1', tokens: 766, family: 'sentencepiece_bpe' },
  { repo: 'NousResearch/Llama-2-7b-hf', tokens: 811, family: 'sentencepiece_bpe' },
  { repo: 'unsloth/gemma-2-2b-it', tokens: 682, family: 'sentencepiece_bpe' },
]

describe('analyzeTokenizer against the reference tokenizers library', () => {
  it.each(REFERENCES)('$repo tokenises the corpus into $tokens tokens', async ({ repo, tokens, family }) => {
    const response = await fetch(`https://huggingface.co/${repo}/resolve/main/tokenizer.json`)
    const json = await response.json()
    const profile = analyzeTokenizer(json)
    expect(profile.family).toBe(family)
    expect(profile.corpusTokens).toBe(tokens)
    expect(profile.corpusBytes).toBe(new TextEncoder().encode(TOKENIZER_CORPUS).length)
  })
})