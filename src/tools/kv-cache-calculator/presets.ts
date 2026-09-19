/**
 * Quick picks for the model id field. These are the most downloaded and most
 * discussed open models at the time of writing, chosen so that every formula
 * family in the engine is represented by something real.
 */
export interface ModelPreset {
  id: string
  note: string
}

export const MODEL_PRESETS: ModelPreset[] = [
  { id: 'Qwen/Qwen3-8B', note: 'Grouped query attention' },
  { id: 'Qwen/Qwen3-32B', note: 'Grouped query attention, 64 layers' },
  { id: 'Qwen/Qwen3-30B-A3B', note: 'MoE, same attention shape as Qwen3' },
  { id: 'Qwen/Qwen2.5-7B-Instruct', note: 'Older but still everywhere' },
  { id: 'Qwen/Qwen3-Next-80B-A3B-Instruct', note: 'Hybrid linear attention' },
  { id: 'Qwen/Qwen3.8-27B', note: 'Hybrid, 262k context' },
  { id: 'Qwen/Qwen3.6-35B-A3B', note: 'Hybrid MoE' },
  { id: 'deepseek-ai/DeepSeek-R1', note: 'MLA, 61 layers' },
  { id: 'deepseek-ai/DeepSeek-V3.2-Exp', note: 'MLA plus an indexer cache' },
  { id: 'deepseek-ai/DeepSeek-V4-Flash', note: 'Compressed sparse attention' },
  { id: 'deepseek-ai/DeepSeek-V4-Pro', note: 'Compressed sparse attention, 61 layers' },
  { id: 'openai/gpt-oss-120b', note: 'Sliding window on half its layers' },
  { id: 'openai/gpt-oss-20b', note: 'Sliding window, smaller sibling' },
  { id: 'google/gemma-4-31b-it', note: 'Shares one vector for key and value' },
  { id: 'zai-org/GLM-4.7-Flash', note: 'MLA, 47 layers' },
  { id: 'zai-org/GLM-5.3', note: 'MLA with a shared indexer cache' },
  { id: 'moonshotai/Kimi-K2-Instruct', note: 'MLA, 128k context' },
  { id: 'mistralai/Mistral-Small-3.1-24B-Instruct-2503', note: 'Standard attention' },
  { id: 'openbmb/MiniCPM5-2B', note: 'Small and quick to reason about' },
  { id: 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', note: 'Hybrid Mamba and attention' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', note: 'Gated. Needs a token.' },
  { id: 'meta-llama/Llama-3.2-3B-Instruct', note: 'Gated. Needs a token.' },
]
