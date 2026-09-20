import type { ModelPreset } from '@/components/model-source/ModelSourceFields'

/**
 * Quick picks for the model id field.
 *
 * Every entry is a real published model, so the suggestion list doubles as a
 * worked example of what the calculator answers for. The notes describe the
 * size the model has to hold, which is the figure that decides the card.
 *
 * Gated repositories are marked, because they need a token before the config
 * can be read at all.
 */
export const MODEL_PRESETS: ModelPreset[] = [
  { id: 'Qwen/Qwen3-8B', note: '8B dense. One 24 GB card in BF16.' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', note: '8B dense. Gated. Needs a token.' },
  { id: 'openai/gpt-oss-120b', note: '117B with 5.1B active. A large MoE.' },
  { id: 'zai-org/GLM-4.7', note: 'Grouped query attention, 92 layers.' },
  { id: 'moonshotai/Kimi-K2', note: 'MLA, 61 layers. A very large MoE.' },
  { id: 'deepseek-ai/DeepSeek-V3.2', note: 'Compressed sparse attention.' },
  { id: 'mistralai/Mistral-Small-3.1-24B-Instruct-2503', note: '24B dense, 40 layers.' },
  { id: 'google/gemma-3-27b-it', note: '27B dense, sliding window attention.' },
  { id: 'Qwen/Qwen3-30B-A3B', note: '30B with 3B active. A sparse MoE.' },
  { id: 'meta-llama/Llama-3.3-70B-Instruct', note: '70B dense. Gated. Needs a token.' },
]
