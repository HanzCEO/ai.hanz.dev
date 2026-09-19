/**
 * Quick picks for the model id field.
 *
 * The first block is the top 50 of the GPQA leaderboard in rank order, so the
 * models people benchmark against are one keystroke away. FINAL-Bench/Darwin-60B-DUO
 * is deliberately absent: it declares transformers_compatible false and is a
 * gateway that routes to two other models, so it has no KV cache of its own.
 * FINAL-Bench/Darwin-27B-Opus is listed but gated, so its shape is not verified.
 *
 * The second block keeps a few models that cover shapes the GPQA list does not:
 * sliding window attention, a shared key and value vector, plain dense attention,
 * and the gated Meta releases.
 *
 * Notes describe the cache shape the engine detects, not the weights.
 */
export interface ModelPreset {
  id: string
  note: string
}

export const MODEL_PRESETS: ModelPreset[] = [
  // GPQA leaderboard, rank order.
  { id: 'moonshotai/Kimi-K3', note: 'GPQA #1 · Hybrid KDA, 24 of 93 layers keep a latent cache' },
  { id: 'ornith-ai/Ornith-1.5-397B', note: 'GPQA #2 · Hybrid linear MoE, 15 of 60 layers cache attention' },
  { id: 'Qwen/Qwen3.8-2.4T-A95B', note: 'GPQA #3 · Hybrid linear MoE, 23 of 92 layers cache attention' },
  { id: 'tencent/Hy4-preview', note: 'GPQA #4 · MLA with a sparse indexer cache, 78 layers' },
  { id: 'Qwen/Qwen3.8-Flash-Next', note: 'GPQA #5 · Hybrid linear, 12 of 48 layers cache attention' },
  { id: 'zai-org/GLM-5.2', note: 'GPQA #6 · MLA with a sparse indexer cache, 78 layers' },
  { id: 'FINAL-Bench/Darwin-398B-JGOS', note: 'GPQA #7 · Hybrid linear MoE, 15 of 60 layers cache attention' },
  { id: 'deepseek-ai/DeepSeek-V4.1-Flash', note: 'GPQA #8 · Compressed sparse attention, 40 layers' },
  { id: 'moonshotai/Kimi-K2.6', note: 'GPQA #9 · MLA, 61 layers' },
  { id: 'tencent/Hy3', note: 'GPQA #10 · Grouped query attention, 80 layers' },
  { id: 'deepseek-ai/DeepSeek-V4-Pro', note: 'GPQA #11 · Compressed sparse attention, 61 layers' },
  { id: 'thinkingmachines/Inkling-Small', note: 'GPQA #12 · Grouped query attention, 42 layers' },
  { id: 'FINAL-Bench/Darwin-28B-REASON', note: 'GPQA #13 · Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'Qwen/Qwen3.8-27B', note: 'GPQA #14 · Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'ornith-ai/Ornith-1.5-35B-A3B', note: 'GPQA #15 · Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'Qwen/Qwen3.5-397B-A17B', note: 'GPQA #16 · Hybrid linear MoE, 15 of 60 layers cache attention' },
  { id: 'FINAL-Bench/Darwin-36B-Opus', note: 'GPQA #17 · Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'inclusionAI/Ring-2.6-1T', note: 'GPQA #19 · MLA, 80 layers' },
  { id: 'deepseek-ai/DeepSeek-V4-Flash', note: 'GPQA #20 · Compressed sparse attention, 43 layers' },
  {
    id: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4',
    note: 'GPQA #21 · Hybrid Mamba, 12 of 108 layers cache attention, NVFP4 weights',
  },
  { id: 'zai-org/GLM-4.7-FP8', note: 'GPQA #22 · Grouped query attention, 92 layers, FP8 weights' },
  { id: 'Qwen/Qwen3.6-27B', note: 'GPQA #23 · Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'moonshotai/Kimi-K2.5', note: 'GPQA #24 · MLA, 61 layers' },
  { id: 'tencent/Hy3-preview', note: 'GPQA #26 · Grouped query attention, 80 layers' },
  { id: 'thinkingmachines/Inkling', note: 'GPQA #27 · Grouped query attention, 66 layers' },
  {
    id: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
    note: 'GPQA #28 · Hybrid Mamba, 12 of 108 layers cache attention',
  },
  { id: 'FINAL-Bench/Darwin-27B-Opus', note: 'GPQA #29 · Gated. Needs a token.' },
  { id: 'Qwen/Qwen3.5-122B-A10B', note: 'GPQA #30 · Hybrid linear MoE, 12 of 48 layers cache attention' },
  { id: 'ornith-ai/Ornith-1.5-9B', note: 'GPQA #31 · Hybrid linear, 8 of 32 layers cache attention' },
  { id: 'FINAL-Bench/Ourbox-35B-JGOS', note: 'GPQA #32 · Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'upstage/Solar-Open2-250B', note: 'GPQA #33 · Hybrid KDA, 12 of 48 layers cache attention' },
  { id: 'zai-org/GLM-5.1', note: 'GPQA #34 · MLA with a sparse indexer cache, 78 layers' },
  {
    id: 'INCModel3/GLM-5.1-MXFP4-Mixed-CT-AutoRound',
    note: 'GPQA #35 · MLA with a sparse indexer cache, MXFP4 weights',
  },
  { id: 'zai-org/GLM-5', note: 'GPQA #36 · MLA with a sparse indexer cache, 78 layers' },
  { id: 'Qwen/Qwen3.6-35B-A3B', note: 'GPQA #37 · Hybrid linear MoE, 10 of 40 layers cache attention' },
  { id: 'FINAL-Bench/Darwin-31B-Opus', note: 'GPQA #38 · Sliding window on 50 of 60 layers' },
  { id: 'zai-org/GLM-4.7', note: 'GPQA #39 · Grouped query attention, 92 layers' },
  { id: 'skt/A.X-K2', note: 'GPQA #40 · MLA with a sparse indexer cache, 61 layers' },
  { id: 'Qwen/Qwen3.5-27B', note: 'GPQA #41 · Hybrid linear, 16 of 64 layers cache attention' },
  { id: 'zai-org/GLM-5-FP8', note: 'GPQA #42 · MLA with a sparse indexer cache, FP8 weights' },
  { id: 'MiniMaxAI/MiniMax-M2.5', note: 'GPQA #43 · Grouped query attention, 62 layers' },
  { id: 'FINAL-Bench/Darwin-4B-David', note: 'GPQA #44 · Sliding window on 35 of 42 layers' },
  { id: 'moonshotai/Kimi-K2-Thinking', note: 'GPQA #46 · MLA, 61 layers' },
  { id: 'FINAL-Bench/Darwin-9B-NEG', note: 'GPQA #47 · Hybrid linear, 8 of 32 layers cache attention' },
  { id: 'JGOS-Model/JGOS-31B-Citizen', note: 'GPQA #48 · Sliding window on 50 of 60 layers' },
  {
    id: 'FlagRelease/Darwin-9B-NEG-FINAL-hygon-FlagOS',
    note: 'GPQA #49 · Hybrid linear, 8 of 32 layers cache attention',
  },
  {
    id: 'FlagRelease/Darwin-9B-NEG-ansulev-hygon-FlagOS',
    note: 'GPQA #50 · Hybrid linear, 8 of 32 layers cache attention',
  },

  // Shapes the GPQA list does not cover.
  { id: 'openai/gpt-oss-120b', note: 'Sliding window on 18 of 36 layers' },
  { id: 'openai/gpt-oss-20b', note: 'Sliding window on 12 of 24 layers' },
  { id: 'google/gemma-4-31b-it', note: 'Sliding window, and one shared vector for key and value' },
  { id: 'mistralai/Mistral-Small-3.1-24B-Instruct-2503', note: 'Plain grouped query attention, 40 layers' },
  { id: 'openbmb/MiniCPM5-2B', note: 'Plain grouped query attention, 42 layers' },
  { id: 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', note: 'Hybrid Mamba, 4 of 42 layers cache attention' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', note: 'Gated. Needs a token.' },
  { id: 'meta-llama/Llama-3.2-3B-Instruct', note: 'Gated. Needs a token.' },
]
