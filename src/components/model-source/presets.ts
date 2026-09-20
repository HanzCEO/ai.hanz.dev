/**
 * Quick picks for the model id field.
 *
 * The first block lists frontier open-weight models, strongest first, so the
 * models people actually compare are one keystroke away. FINAL-Bench/Darwin-60B-DUO
 * is deliberately absent: it declares transformers_compatible false and is a
 * gateway that routes to two other models, so it has no KV cache of its own.
 * FINAL-Bench/Darwin-27B-Opus is listed but gated, so its shape is not verified.
 *
 * The second block keeps a few models that cover shapes the first does not:
 * sliding window attention, a shared key and value vector, plain dense attention,
 * and the gated Meta releases.
 *
 * Notes describe the cache shape the engine detects, not the weights.
 */
import type { ModelPreset } from './ModelSourceFields'

export type { ModelPreset }

export const MODEL_PRESETS: ModelPreset[] = [
  // Frontier open-weight models, strongest first.
  { id: 'moonshotai/Kimi-K3', note: 'Hybrid KDA, 24 of 93 layers keep a latent cache' },
  { id: 'ornith-ai/Ornith-1.5-397B', note: 'Hybrid linear MoE, 15 of 60 layers cache attention' },
  { id: 'Qwen/Qwen3.8-2.4T-A95B', note: 'Hybrid linear MoE, 23 of 92 layers cache attention' },
  { id: 'tencent/Hy4-preview', note: 'MLA with a sparse indexer cache, 78 layers' },
  { id: 'Qwen/Qwen3.8-Flash-Next', note: 'Hybrid linear, 12 of 48 layers cache attention' },
  { id: 'zai-org/GLM-5.2', note: 'MLA with a sparse indexer cache, 78 layers' },
  { id: 'FINAL-Bench/Darwin-398B-JGOS', note: 'Hybrid linear MoE, 15 of 60 layers cache attention' },
  {
    id: 'deepseek-ai/DeepSeek-V4.1-Flash',
    note: 'Compressed sparse attention 2, only 4 of 40 layers own a global cache',
  },
  { id: 'moonshotai/Kimi-K2.6', note: 'MLA, 61 layers' },
  { id: 'tencent/Hy3', note: 'Grouped query attention, 80 layers' },
  { id: 'deepseek-ai/DeepSeek-V4-Pro', note: 'Compressed sparse attention, 61 layers' },
  { id: 'thinkingmachines/Inkling-Small', note: 'Grouped query attention, 42 layers' },
  { id: 'FINAL-Bench/Darwin-28B-REASON', note: 'Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'Qwen/Qwen3.8-27B', note: 'Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'ornith-ai/Ornith-1.5-35B-A3B', note: 'Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'Qwen/Qwen3.5-397B-A17B', note: 'Hybrid linear MoE, 15 of 60 layers cache attention' },
  { id: 'FINAL-Bench/Darwin-36B-Opus', note: 'Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'inclusionAI/Ring-2.6-1T', note: 'MLA, 80 layers' },
  { id: 'deepseek-ai/DeepSeek-V4-Flash', note: 'Compressed sparse attention, 43 layers' },
  {
    id: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4',
    note: 'Hybrid Mamba, 12 of 108 layers cache attention, NVFP4 weights',
  },
  { id: 'zai-org/GLM-4.7-FP8', note: 'Grouped query attention, 92 layers, FP8 weights' },
  { id: 'Qwen/Qwen3.6-27B', note: 'Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'moonshotai/Kimi-K2.5', note: 'MLA, 61 layers' },
  { id: 'tencent/Hy3-preview', note: 'Grouped query attention, 80 layers' },
  { id: 'thinkingmachines/Inkling', note: 'Grouped query attention, 66 layers' },
  {
    id: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
    note: 'Hybrid Mamba, 12 of 108 layers cache attention',
  },
  { id: 'FINAL-Bench/Darwin-27B-Opus', note: 'Gated. Needs a token.' },
  { id: 'Qwen/Qwen3.5-122B-A10B', note: 'Hybrid linear MoE, 12 of 48 layers cache attention' },
  { id: 'ornith-ai/Ornith-1.5-9B', note: 'Hybrid linear, 8 of 32 layers cache attention' },
  { id: 'FINAL-Bench/Ourbox-35B-JGOS', note: 'Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'upstage/Solar-Open2-250B', note: 'Hybrid KDA, 12 of 48 layers cache attention' },
  { id: 'zai-org/GLM-5.1', note: 'MLA with a sparse indexer cache, 78 layers' },
  {
    id: 'INCModel3/GLM-5.1-MXFP4-Mixed-CT-AutoRound',
    note: 'MLA with a sparse indexer cache, MXFP4 weights',
  },
  { id: 'zai-org/GLM-5', note: 'MLA with a sparse indexer cache, 78 layers' },
  { id: 'Qwen/Qwen3.6-35B-A3B', note: 'Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'FINAL-Bench/Darwin-31B-Opus', note: 'Sliding window on 50 of 60 layers' },
  { id: 'zai-org/GLM-4.7', note: 'Grouped query attention, 92 layers' },
  { id: 'skt/A.X-K2', note: 'MLA with a sparse indexer cache, 61 layers' },
  { id: 'Qwen/Qwen3.5-27B', note: 'Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'zai-org/GLM-5-FP8', note: 'MLA with a sparse indexer cache, FP8 weights' },
  { id: 'MiniMaxAI/MiniMax-M2.5', note: 'Grouped query attention, 62 layers' },
  { id: 'FINAL-Bench/Darwin-4B-David', note: 'Sliding window on 35 of 42 layers' },
  { id: 'moonshotai/Kimi-K2-Thinking', note: 'MLA, 61 layers' },
  { id: 'FINAL-Bench/Darwin-9B-NEG', note: 'Hybrid linear, 8 of 32 layers cache attention' },
  { id: 'JGOS-Model/JGOS-31B-Citizen', note: 'Sliding window on 50 of 60 layers' },
  {
    id: 'FlagRelease/Darwin-9B-NEG-FINAL-hygon-FlagOS',
    note: 'Hybrid linear, 8 of 32 layers cache attention',
  },
  {
    id: 'FlagRelease/Darwin-9B-NEG-ansulev-hygon-FlagOS',
    note: 'Hybrid linear, 8 of 32 layers cache attention',
  },

  // Shapes the first block does not cover.
  { id: 'openai/gpt-oss-120b', note: 'Sliding window on 18 of 36 layers' },
  { id: 'openai/gpt-oss-20b', note: 'Sliding window on 12 of 24 layers' },
  { id: 'google/gemma-4-31b-it', note: 'Sliding window, and one shared vector for key and value' },
  { id: 'mistralai/Mistral-Small-3.1-24B-Instruct-2503', note: 'Plain grouped query attention, 40 layers' },
  { id: 'openbmb/MiniCPM5-2B', note: 'Plain grouped query attention, 42 layers' },
  { id: 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', note: 'Hybrid Mamba, 4 of 42 layers cache attention' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', note: 'Gated. Needs a token.' },
  { id: 'meta-llama/Llama-3.2-3B-Instruct', note: 'Gated. Needs a token.' },
]
