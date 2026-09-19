import { DEFAULT_DTYPE, DTYPES, dtypeBytes, dtypeSupport, isDtypeId } from './dtypes'
import {
  KvCacheInputError,
  type ArchitectureFamily,
  type ComponentBreakdown,
  type ComputeOptions,
  type ComputeResult,
  type ConstantUsed,
  type DtypeSupport,
  type LayerSplit,
  type RawConfig,
} from './types'

// ---------------------------------------------------------------------------
// Config field readers. Configs in the wild omit fields freely, so everything
// is read defensively and absence is handled rather than assumed away.
// ---------------------------------------------------------------------------

const NESTED_KEYS = ['text_config', 'llm_config', 'language_config'] as const

function readNumber(config: RawConfig, key: string): number | undefined {
  const value = config[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
    return Number(value)
  }
  return undefined
}

function readNumberArray(config: RawConfig, key: string): number[] | undefined {
  const value = config[key]
  if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'number')) {
    return value as number[]
  }
  return undefined
}

function readStringArray(config: RawConfig, key: string): string[] | undefined {
  const value = config[key]
  if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'string')) {
    return value as string[]
  }
  return undefined
}

function readString(config: RawConfig, key: string): string | undefined {
  const value = config[key]
  return typeof value === 'string' ? value : undefined
}

function readBoolean(config: RawConfig, key: string): boolean | undefined {
  const value = config[key]
  return typeof value === 'boolean' ? value : undefined
}

/**
 * Multimodal configs (DeepSeek-V4.1-Flash, Qwen3.5, Gemma 4) keep the language
 * model fields under a nested key. Returns the inner config plus the outer one,
 * since a few fields such as quantization_config only live outside.
 */
export function unwrapConfig(raw: RawConfig): { inner: RawConfig; outer: RawConfig } {
  for (const key of NESTED_KEYS) {
    const nested = raw[key]
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      return { inner: nested as RawConfig, outer: raw }
    }
  }
  return { inner: raw, outer: raw }
}

/** Converts a torch dtype name into bytes per element. */
function dtypeNameToBytes(name: string | undefined): number | undefined {
  if (!name) return undefined
  const normalized = name.toLowerCase()
  if (normalized.includes('float32') || normalized === 'fp32') return 4
  if (normalized.includes('float16') || normalized === 'fp16' || normalized === 'half') return 2
  if (normalized.includes('bfloat16') || normalized === 'bf16') return 2
  if (normalized.includes('float8') || normalized === 'fp8') return 1
  if (normalized.includes('int8')) return 1
  return undefined
}

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

const DSV4_TYPES = new Set(['deepseek_v4', 'deepseek_v41', 'deepseek_v4_text', 'deepseek_v41_text'])
const HYBRID_TYPES = new Set([
  'qwen3_next',
  'qwen3_5',
  'qwen3_5_moe',
  'qwen3_5_text',
  'qwen3_5_moe_text',
  'nemotron_h',
  'bamba',
  'zamba2',
  'jamba',
  'lfm2',
  'falcon_h1',
  'granitemoehybrid',
])
const MLA_TYPES = new Set([
  'deepseek_v2',
  'deepseek_v3',
  'deepseek_v32',
  'kimi_k2',
  'glm4_moe_lite',
  'glm_moe_dsa',
  'xing4_0',
  'ernie4_5_moe',
])
const GQA_TYPES = new Set([
  'qwen2',
  'qwen3',
  'qwen3_moe',
  'qwen2_moe',
  'qwen3_5_dense',
  'llama',
  'mistral',
  'mixtral',
  'gemma2',
  'gemma3',
  'gemma4',
  'gemma4_text',
  'gpt_oss',
  'phi3',
  'phi4',
  'olmo2',
  'glm4',
  'glm4_moe',
  'internlm2',
  'starcoder2',
  'cohere',
  'cohere2',
  'granite',
  'exaone',
  'minicpm',
  'smollm',
  'smollm3',
  'yi',
  'deepseek',
  'bloom',
  'opt',
  'falcon',
  'nemotron',
  'hunyuan_v1_dense',
  'hunyuan_v1_moe',
])

const FAMILY_LABELS: Record<ArchitectureFamily, string> = {
  gqa: 'Grouped query attention',
  mla: 'Multi head latent attention',
  hybrid_linear: 'Hybrid linear attention',
  dsv4: 'Compressed sparse attention (DeepSeek V4)',
  unknown: 'Unrecognised architecture',
}

export interface Detection {
  family: ArchitectureFamily
  modelType: string
  label: string
  bestEffort: boolean
  reason: string
}

export function detectArchitecture(inner: RawConfig, outer: RawConfig): Detection {
  const modelType =
    readString(outer, 'model_type') ?? readString(inner, 'model_type') ?? 'unknown'
  const architectures = readStringArray(outer, 'architectures') ?? []
  const candidates = [modelType, ...architectures].map((value) => value.toLowerCase())

  const matches = (set: Set<string>) => candidates.some((value) => set.has(value))

  if (matches(DSV4_TYPES) || readNumberArray(inner, 'compress_ratios')) {
    return {
      family: 'dsv4',
      modelType,
      label: FAMILY_LABELS.dsv4,
      bestEffort: !matches(DSV4_TYPES),
      reason: 'Config carries per layer compress_ratios.',
    }
  }

  if (
    matches(HYBRID_TYPES) ||
    readNumber(inner, 'linear_num_value_heads') !== undefined ||
    readString(inner, 'hybrid_override_pattern') !== undefined
  ) {
    return {
      family: 'hybrid_linear',
      modelType,
      label: FAMILY_LABELS.hybrid_linear,
      bestEffort: !matches(HYBRID_TYPES),
      reason: 'Config mixes linear attention layers with periodic full attention.',
    }
  }

  if (matches(MLA_TYPES) || readNumber(inner, 'kv_lora_rank') !== undefined) {
    return {
      family: 'mla',
      modelType,
      label: FAMILY_LABELS.mla,
      bestEffort: !matches(MLA_TYPES),
      reason: 'Config has kv_lora_rank, so the cache is a compressed latent.',
    }
  }

  if (matches(GQA_TYPES) || readNumber(inner, 'num_key_value_heads') !== undefined) {
    return {
      family: 'gqa',
      modelType,
      label: FAMILY_LABELS.gqa,
      bestEffort: !matches(GQA_TYPES),
      reason: 'Config has num_key_value_heads, so the cache is per key/value head.',
    }
  }

  return {
    family: 'unknown',
    modelType,
    label: FAMILY_LABELS.unknown,
    bestEffort: true,
    reason: 'No attention fields were recognised in the config.',
  }
}

// ---------------------------------------------------------------------------
// Shared computation scaffolding
// ---------------------------------------------------------------------------

interface LayerKind {
  label: string
  /** Bytes for one cached entry in one layer. */
  bytesPerEntry: number
  /** Cached entries for one sequence. */
  entriesPerSequence: number
  /** Number of layers of this kind. */
  layers: number
  /** Formula text for the breakdown. */
  formula: string
}

interface FamilyComputation {
  family: ArchitectureFamily
  layerSplit: LayerSplit
  attention: { kinds: LayerKind[]; note?: string }
  indexer: { bytesPerEntry: number; entriesPerSequence: number; layers: number; formula: string; note?: string } | null
  state: { bytesPerSequence: number; formula: string; inputs: Array<{ key: string; value: number | string }>; note?: string } | null
  constants: ConstantUsed[]
  assumptions: string[]
  bestEffort: boolean
  bestEffortReason?: string
  steps: Array<{ label: string; detail: string }>
}

function countBy(layerTypes: string[], wanted: string): number {
  return layerTypes.filter((type) => type === wanted).length
}

function sumKinds(kinds: LayerKind[]): number {
  return kinds.reduce((total, kind) => total + kind.layers * kind.bytesPerEntry * kind.entriesPerSequence, 0)
}

function countCompressed(ratios: number[]): Array<{ ratio: number; layers: number }> {
  const counts = new Map<number, number>()
  for (const ratio of ratios) counts.set(ratio, (counts.get(ratio) ?? 0) + 1)
  return Array.from(counts.entries())
    .map(([ratio, layers]) => ({ ratio, layers }))
    .sort((a, b) => a.ratio - b.ratio)
}

// ---------------------------------------------------------------------------
// Family formulas
// ---------------------------------------------------------------------------

function computeGqa(
  inner: RawConfig,
  options: ComputeOptions,
  kvBytes: number,
): FamilyComputation {
  const contextLength = options.contextLength
  const constants: ConstantUsed[] = []
  const assumptions: string[] = []

  const numLayers = readNumber(inner, 'num_hidden_layers')
  const numHeads = readNumber(inner, 'num_attention_heads')
  const hiddenSize = readNumber(inner, 'hidden_size')

  if (!numLayers || !numHeads) {
    throw new KvCacheInputError(
      'This config does not list num_hidden_layers and num_attention_heads, so the cache size cannot be derived from it.',
    )
  }

  const kvHeads = readNumber(inner, 'num_key_value_heads') ?? numHeads
  const headDim = readNumber(inner, 'head_dim') ?? Math.floor((hiddenSize ?? 0) / numHeads)

  if (!headDim || headDim <= 0) {
    throw new KvCacheInputError(
      'Could not determine head_dim. The config has neither head_dim nor hidden_size and num_attention_heads.',
    )
  }

  constants.push(
    { key: 'num_hidden_layers', value: numLayers, source: 'config' },
    { key: 'num_attention_heads', value: numHeads, source: 'config' },
    {
      key: 'num_key_value_heads',
      value: kvHeads,
      source: readNumber(inner, 'num_key_value_heads') === undefined ? 'assumed equal to num_attention_heads' : 'config',
    },
    {
      key: 'head_dim',
      value: headDim,
      source: readNumber(inner, 'head_dim') === undefined ? 'derived from hidden_size / num_attention_heads' : 'config',
    },
  )

  // Some architectures store one vector and reuse it for both key and value.
  const kEqV = readBoolean(inner, 'attention_k_eq_v') === true
  const kvMultiplier = kEqV ? 1 : 2
  if (kEqV) {
    assumptions.push(
      'attention_k_eq_v is set, so key and value share one vector and the cache is counted once per layer rather than twice.',
    )
  }

  const layerTypes = readStringArray(inner, 'layer_types')
  const slidingWindow = readNumber(inner, 'sliding_window')
  const globalHeadDim = readNumber(inner, 'global_head_dim')
  const globalKvHeads = readNumber(inner, 'num_global_key_value_heads')

  const kinds: LayerKind[] = []

  if (layerTypes && layerTypes.length >= numLayers) {
    const types = layerTypes.slice(0, numLayers)
    const slidingCount = countBy(types, 'sliding_attention')
    const fullCount = numLayers - slidingCount

    if (fullCount > 0) {
      // Full attention layers can use a wider head and fewer key/value heads.
      const fullHeadDim = globalHeadDim ?? headDim
      const fullKvHeads = globalKvHeads ?? kvHeads
      kinds.push({
        label: 'full attention layers',
        bytesPerEntry: kvMultiplier * fullKvHeads * fullHeadDim * kvBytes,
        entriesPerSequence: contextLength,
        layers: fullCount,
        formula: `${kvMultiplier} x ${fullKvHeads} kv heads x ${fullHeadDim} head dim x ${kvBytes} bytes`,
      })
      if (globalHeadDim || globalKvHeads) {
        constants.push(
          { key: 'global_head_dim', value: fullHeadDim, source: globalHeadDim ? 'config' : 'assumed equal to head_dim' },
          {
            key: 'num_global_key_value_heads',
            value: fullKvHeads,
            source: globalKvHeads ? 'config' : 'assumed equal to num_key_value_heads',
          },
        )
        assumptions.push(
          'Full attention layers use global_head_dim and num_global_key_value_heads, which differ from the sliding layers.',
        )
      }
    }

    if (slidingCount > 0) {
      if (!slidingWindow || slidingWindow <= 0) {
        assumptions.push(
          'layer_types marks layers as sliding but the config has no sliding_window, so those layers were counted at the full context length.',
        )
      }
      const window = slidingWindow && slidingWindow > 0 ? slidingWindow : contextLength
      const entries = Math.min(contextLength, window)
      kinds.push({
        label: 'sliding window layers',
        bytesPerEntry: kvMultiplier * kvHeads * headDim * kvBytes,
        entriesPerSequence: entries,
        layers: slidingCount,
        formula: `${kvMultiplier} x ${kvHeads} kv heads x ${headDim} head dim x ${kvBytes} bytes`,
      })
      constants.push({ key: 'sliding_window', value: window, source: slidingWindow ? 'config' : 'assumed equal to context length' })
      if (entries < contextLength) {
        assumptions.push(
          `Sliding window layers hold at most ${window} entries per sequence, so they stop growing past that point.`,
        )
      }
    }
  } else if (slidingWindow && slidingWindow > 0) {
    // No per layer types, but the model applies one window everywhere.
    const entries = Math.min(contextLength, slidingWindow)
    kinds.push({
      label: 'sliding window layers',
      bytesPerEntry: kvMultiplier * kvHeads * headDim * kvBytes,
      entriesPerSequence: entries,
      layers: numLayers,
      formula: `${kvMultiplier} x ${kvHeads} kv heads x ${headDim} head dim x ${kvBytes} bytes`,
    })
    constants.push({ key: 'sliding_window', value: slidingWindow, source: 'config' })
    assumptions.push(
      `Every layer uses a sliding window of ${slidingWindow}, so the cache stops growing past that length.`,
    )
  } else {
    kinds.push({
      label: 'attention layers',
      bytesPerEntry: kvMultiplier * kvHeads * headDim * kvBytes,
      entriesPerSequence: contextLength,
      layers: numLayers,
      formula: `${kvMultiplier} x ${kvHeads} kv heads x ${headDim} head dim x ${kvBytes} bytes`,
    })
  }

  const fullAttention = kinds.find((kind) => kind.label.startsWith('full'))?.layers ?? 0
  const slidingAttention = kinds.find((kind) => kind.label.startsWith('sliding'))?.layers ?? 0
  const plainAttention = kinds.find((kind) => kind.label === 'attention layers')?.layers ?? 0

  return {
    family: 'gqa',
    layerSplit: {
      total: numLayers,
      fullAttention: fullAttention + plainAttention,
      slidingAttention,
      linearAttention: 0,
    },
    attention: { kinds },
    indexer: null,
    state: null,
    constants,
    assumptions,
    bestEffort: false,
    steps: [],
  }
}

function computeMla(
  inner: RawConfig,
  options: ComputeOptions,
  kvBytes: number,
  indexerBytes: number,
): FamilyComputation {
  const contextLength = options.contextLength
  const constants: ConstantUsed[] = []
  const assumptions: string[] = []

  const numLayers = readNumber(inner, 'num_hidden_layers')
  const kvLoraRank = readNumber(inner, 'kv_lora_rank')
  const ropeDim = readNumber(inner, 'qk_rope_head_dim') ?? 64

  if (!numLayers || !kvLoraRank) {
    throw new KvCacheInputError(
      'This config does not list num_hidden_layers and kv_lora_rank, so the latent cache size cannot be derived from it.',
    )
  }

  constants.push(
    { key: 'num_hidden_layers', value: numLayers, source: 'config' },
    { key: 'kv_lora_rank', value: kvLoraRank, source: 'config' },
    {
      key: 'qk_rope_head_dim',
      value: ropeDim,
      source: readNumber(inner, 'qk_rope_head_dim') === undefined ? 'assumed 64' : 'config',
    },
  )

  const bytesPerEntry = (kvLoraRank + ropeDim) * kvBytes

  assumptions.push(
    'MLA stores one compressed latent per token per layer, so the cache does not scale with the number of attention heads.',
  )

  const kinds: LayerKind[] = [
    {
      label: 'latent attention layers',
      bytesPerEntry,
      entriesPerSequence: contextLength,
      layers: numLayers,
      formula: `(${kvLoraRank} kv_lora_rank + ${ropeDim} qk_rope_head_dim) x ${kvBytes} bytes`,
    },
  ]

  // Indexer cache, present on DeepSeek-V3.2 style sparse attention models.
  let indexer: FamilyComputation['indexer'] = null
  const indexHeadDim = readNumber(inner, 'index_head_dim')
  const indexNHeads = readNumber(inner, 'index_n_heads')

  if (indexHeadDim && indexNHeads) {
    const indexerTypes = readStringArray(inner, 'indexer_types')
    let indexerLayers = numLayers
    let note = 'The indexer keeps one key per token per layer, so it does not scale with index_n_heads.'

    if (indexerTypes && indexerTypes.length >= numLayers) {
      const own = countBy(indexerTypes.slice(0, numLayers), 'full')
      if (own > 0 && own < numLayers) {
        indexerLayers = own
        note = `Only ${own} of ${numLayers} layers store their own indexer cache. The rest share another layer's, per indexer_types.`
        constants.push({ key: 'indexer_types', value: `${own} full / ${numLayers - own} shared`, source: 'config' })
      }
    }

    indexer = {
      bytesPerEntry: indexHeadDim * indexerBytes,
      entriesPerSequence: contextLength,
      layers: indexerLayers,
      formula: `${indexHeadDim} index_head_dim x ${indexerBytes} bytes`,
      note,
    }

    constants.push(
      { key: 'index_head_dim', value: indexHeadDim, source: 'config' },
      { key: 'index_n_heads', value: indexNHeads, source: 'config' },
    )

    assumptions.push(
      'The indexer key cache is counted separately from the latent cache, which is what DeepSeek sparse attention adds.',
    )
  }

  return {
    family: 'mla',
    layerSplit: {
      total: numLayers,
      fullAttention: numLayers,
      slidingAttention: 0,
      linearAttention: 0,
    },
    attention: { kinds },
    indexer,
    state: null,
    constants,
    assumptions,
    bestEffort: false,
    steps: [],
  }
}

function parseHybridPattern(pattern: string): { attention: number; linear: number } {
  let attention = 0
  let linear = 0
  for (const char of pattern) {
    if (char === '*') attention += 1
    else if (char === 'M' || char === 'm' || char === 'L') linear += 1
    // '-' marks a layer with no attention at all, so it caches nothing.
  }
  return { attention, linear }
}

function computeHybrid(
  inner: RawConfig,
  options: ComputeOptions,
  kvBytes: number,
): FamilyComputation {
  const contextLength = options.contextLength
  const constants: ConstantUsed[] = []
  const assumptions: string[] = []

  const numLayers = readNumber(inner, 'num_hidden_layers')
  const numHeads = readNumber(inner, 'num_attention_heads')
  const hiddenSize = readNumber(inner, 'hidden_size')

  if (!numLayers || !numHeads) {
    throw new KvCacheInputError(
      'This config does not list num_hidden_layers and num_attention_heads, so the cache size cannot be derived from it.',
    )
  }

  const kvHeads = readNumber(inner, 'num_key_value_heads') ?? numHeads
  const headDim = readNumber(inner, 'head_dim') ?? Math.floor((hiddenSize ?? 0) / numHeads)
  if (!headDim || headDim <= 0) {
    throw new KvCacheInputError('Could not determine head_dim for the full attention layers.')
  }

  constants.push(
    { key: 'num_hidden_layers', value: numLayers, source: 'config' },
    { key: 'num_key_value_heads', value: kvHeads, source: 'config' },
    { key: 'head_dim', value: headDim, source: readNumber(inner, 'head_dim') ? 'config' : 'derived' },
  )

  const layerTypes = readStringArray(inner, 'layer_types')
  const fullAttentionInterval = readNumber(inner, 'full_attention_interval')
  const hybridPattern = readString(inner, 'hybrid_override_pattern')

  let fullLayers = 0
  let linearLayers = 0

  if (layerTypes && layerTypes.length >= numLayers) {
    const types = layerTypes.slice(0, numLayers)
    linearLayers = countBy(types, 'linear_attention')
    fullLayers = numLayers - linearLayers
    constants.push({ key: 'layer_types', value: `${fullLayers} full / ${linearLayers} linear`, source: 'config' })
  } else if (hybridPattern) {
    const parsed = parseHybridPattern(hybridPattern)
    fullLayers = parsed.attention
    linearLayers = parsed.linear
    constants.push({ key: 'hybrid_override_pattern', value: hybridPattern, source: 'config' })
    assumptions.push(
      'Layer roles were read from hybrid_override_pattern. Layers marked with a dash hold no attention cache at all.',
    )
  } else if (fullAttentionInterval && fullAttentionInterval > 0) {
    fullLayers = Math.floor(numLayers / fullAttentionInterval)
    linearLayers = numLayers - fullLayers
    constants.push({ key: 'full_attention_interval', value: fullAttentionInterval, source: 'config' })
  } else {
    throw new KvCacheInputError(
      'This hybrid config has no layer_types, hybrid_override_pattern, or full_attention_interval, so the layer split is unknown.',
    )
  }

  const slidingWindow = readNumber(inner, 'sliding_window')
  const kinds: LayerKind[] = []

  if (fullLayers > 0) {
    const window = slidingWindow && slidingWindow > 0 ? slidingWindow : contextLength
    kinds.push({
      label: 'full attention layers',
      bytesPerEntry: 2 * kvHeads * headDim * kvBytes,
      entriesPerSequence: Math.min(contextLength, window),
      layers: fullLayers,
      formula: `2 x ${kvHeads} kv heads x ${headDim} head dim x ${kvBytes} bytes`,
    })
  }

  // Linear attention layers hold a fixed size recurrent state. It does not grow
  // with context length, which is the whole point of the architecture.
  const stateDtypeName = readString(inner, 'mamba_ssm_dtype')
  const stateBytes = dtypeNameToBytes(stateDtypeName) ?? kvBytes
  if (stateDtypeName && dtypeNameToBytes(stateDtypeName) === undefined) {
    assumptions.push(
      `mamba_ssm_dtype is "${stateDtypeName}", which was not recognised, so the state was counted at the attention cache width.`,
    )
  }

  const linearValueHeads = readNumber(inner, 'linear_num_value_heads')
  const linearKeyHeads = readNumber(inner, 'linear_num_key_heads')
  const linearKeyHeadDim = readNumber(inner, 'linear_key_head_dim')
  const linearValueHeadDim = readNumber(inner, 'linear_value_head_dim')
  const linearConvKernel = readNumber(inner, 'linear_conv_kernel_dim')

  const mambaHeads = readNumber(inner, 'mamba_num_heads')
  const mambaHeadDim = readNumber(inner, 'mamba_head_dim')
  const ssmStateSize = readNumber(inner, 'ssm_state_size')
  const nGroups = readNumber(inner, 'n_groups')
  const convKernel = readNumber(inner, 'conv_kernel')

  let stateElementsPerLayer: number | undefined
  let stateFormula = ''
  const stateInputs: Array<{ key: string; value: number | string }> = []

  if (linearValueHeads && linearKeyHeadDim && linearValueHeadDim && linearKeyHeads) {
    const recurrent = linearValueHeads * linearKeyHeadDim * linearValueHeadDim
    const conv =
      (linearKeyHeads * linearKeyHeadDim + 2 * linearValueHeads * linearValueHeadDim) *
      (linearConvKernel ?? 4)
    stateElementsPerLayer = recurrent + conv
    stateFormula = `${linearValueHeads} value heads x ${linearKeyHeadDim} key dim x ${linearValueHeadDim} value dim, plus a conv state of width ${linearConvKernel ?? 4}`
    stateInputs.push(
      { key: 'linear_num_value_heads', value: linearValueHeads },
      { key: 'linear_num_key_heads', value: linearKeyHeads },
      { key: 'linear_key_head_dim', value: linearKeyHeadDim },
      { key: 'linear_value_head_dim', value: linearValueHeadDim },
      { key: 'linear_conv_kernel_dim', value: linearConvKernel ?? 4 },
    )
    constants.push(
      { key: 'linear_num_value_heads', value: linearValueHeads, source: 'config' },
      { key: 'linear_key_head_dim', value: linearKeyHeadDim, source: 'config' },
      { key: 'linear_value_head_dim', value: linearValueHeadDim, source: 'config' },
    )
  } else if (mambaHeads && mambaHeadDim && ssmStateSize) {
    const recurrent = mambaHeads * mambaHeadDim * ssmStateSize
    const conv =
      (mambaHeads * mambaHeadDim + 2 * (nGroups ?? 1) * ssmStateSize) * (convKernel ?? 4)
    stateElementsPerLayer = recurrent + conv
    stateFormula = `${mambaHeads} mamba heads x ${mambaHeadDim} head dim x ${ssmStateSize} state size, plus a conv state of width ${convKernel ?? 4}`
    stateInputs.push(
      { key: 'mamba_num_heads', value: mambaHeads },
      { key: 'mamba_head_dim', value: mambaHeadDim },
      { key: 'ssm_state_size', value: ssmStateSize },
    )
    constants.push(
      { key: 'mamba_num_heads', value: mambaHeads, source: 'config' },
      { key: 'mamba_head_dim', value: mambaHeadDim, source: 'config' },
      { key: 'ssm_state_size', value: ssmStateSize, source: 'config' },
    )
    assumptions.push(
      'The Mamba state was sized from mamba_num_heads, mamba_head_dim, and ssm_state_size. Verify against the implementation you deploy.',
    )
  }

  let state: FamilyComputation['state'] = null
  if (stateElementsPerLayer && linearLayers > 0) {
    const bytesPerSequence = stateElementsPerLayer * linearLayers * stateBytes
    state = {
      bytesPerSequence,
      formula: `${linearLayers} linear layers x (${stateFormula}) x ${stateBytes} bytes`,
      inputs: stateInputs,
      note: 'Constant. It does not grow with context length, so it is reported per sequence rather than per token.',
    }
    assumptions.push(
      stateDtypeName
        ? `Linear attention state was counted at ${stateBytes} bytes per element, from mamba_ssm_dtype.`
        : `The config does not set mamba_ssm_dtype, so the linear attention state was counted at the attention cache width of ${stateBytes} bytes per element.`,
    )
  } else if (linearLayers > 0) {
    assumptions.push(
      'This config has linear attention layers but no fields describing their state size, so the constant recurrent state was left out of the total.',
    )
  }

  return {
    family: 'hybrid_linear',
    layerSplit: {
      total: numLayers,
      fullAttention: fullLayers,
      slidingAttention: 0,
      linearAttention: linearLayers,
    },
    attention: { kinds },
    indexer: null,
    state,
    constants,
    assumptions,
    bestEffort: !layerTypes && Boolean(hybridPattern),
    bestEffortReason: hybridPattern
      ? 'The layer split came from hybrid_override_pattern rather than layer_types, so treat it as a best effort estimate.'
      : undefined,
    steps: [],
  }
}

function computeDsv4(
  inner: RawConfig,
  options: ComputeOptions,
  kvBytes: number,
  indexerBytes: number,
): FamilyComputation {
  const contextLength = options.contextLength
  const constants: ConstantUsed[] = []
  const assumptions: string[] = []

  const numLayers = readNumber(inner, 'num_hidden_layers')
  const headDim = readNumber(inner, 'head_dim')
  const ropeDim = readNumber(inner, 'qk_rope_head_dim')
  const ratios = readNumberArray(inner, 'compress_ratios')

  if (!numLayers || !headDim || !ratios) {
    throw new KvCacheInputError(
      'This DeepSeek V4 config is missing num_hidden_layers, head_dim, or compress_ratios, so the compressed cache size cannot be derived from it.',
    )
  }

  const rope = ropeDim ?? 64
  // The array is one longer than the layer count in current configs because it
  // also covers the multi token prediction layer, which is not part of the
  // running model's cache.
  const usable = ratios.slice(0, numLayers)
  if (ratios.length > numLayers) {
    assumptions.push(
      `compress_ratios has ${ratios.length} entries for ${numLayers} layers. Only the first ${numLayers} were used, since the extra entry covers the multi token prediction layer.`,
    )
  }

  const sharedKvElements = headDim + rope
  const indexHeadDim = readNumber(inner, 'index_head_dim') ?? 0

  constants.push(
    { key: 'num_hidden_layers', value: numLayers, source: 'config' },
    { key: 'head_dim', value: headDim, source: 'config' },
    { key: 'qk_rope_head_dim', value: rope, source: ropeDim ? 'config' : 'assumed 64' },
    { key: 'compress_ratios', value: countCompressed(usable).map((entry) => `${entry.layers}x ratio ${entry.ratio}`).join(', '), source: 'config' },
  )

  assumptions.push(
    'DeepSeek V4 shares one key/value vector per layer, so the cache is counted once per layer rather than twice.',
  )

  const hasIndexer = indexHeadDim > 0 && (readNumber(inner, 'index_n_heads') ?? 0) > 0
  if (hasIndexer) {
    constants.push({ key: 'index_head_dim', value: indexHeadDim, source: 'config' })
    assumptions.push(
      'The indexer key cache is compressed with the same ratio as the attention cache, so it shrinks along with it.',
    )
    if (indexerBytes >= 2) {
      assumptions.push(
        'DeepSeek runs its indexer in FP8 precision. With the indexer at this width the result lands about 11 percent above the published 9.62 GiB for a 61 layer V4 at 1M context. Setting the indexer dtype to FP8 (E4M3) brings it within about 1 percent.',
      )
    }
  }

  const kinds: LayerKind[] = []
  for (const entry of countCompressed(usable)) {
    const divisor = Math.max(1, entry.ratio)
    const bytesPerEntry = (sharedKvElements * kvBytes + (hasIndexer ? indexHeadDim * indexerBytes : 0)) / divisor
    const label = entry.ratio <= 1 ? 'uncompressed layers' : `compressed layers (ratio ${entry.ratio})`
    const parts = [`${sharedKvElements} shared kv elements x ${kvBytes} bytes`]
    if (hasIndexer) parts.push(`${indexHeadDim} indexer elements x ${indexerBytes} bytes`)
    kinds.push({
      label,
      bytesPerEntry,
      entriesPerSequence: contextLength,
      layers: entry.layers,
      formula: entry.ratio <= 1 ? parts.join(' + ') : `(${parts.join(' + ')}) / ${divisor}`,
    })
  }

  if (indexHeadDim === 0) {
    assumptions.push(
      'No index_head_dim in the config, so the indexer cache was left out. V4 deployments normally keep one.',
    )
  }

  return {
    family: 'dsv4',
    layerSplit: {
      total: numLayers,
      fullAttention: 0,
      slidingAttention: 0,
      linearAttention: 0,
      compressed: countCompressed(usable),
    },
    attention: { kinds, note: 'Each layer stores one shared key/value vector per token, divided by its compression ratio.' },
    indexer: null,
    state: null,
    constants,
    assumptions,
    bestEffort: false,
    steps: [],
  }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

function validate(options: ComputeOptions): void {
  const { contextLength, sequenceCount } = options

  for (const [name, dtype] of [
    ['kv_cache_dtype', options.kvCacheDtype],
    ['indexer_dtype', options.indexerDtype],
  ] as const) {
    if (!isDtypeId(dtype)) {
      const known = DTYPES.map((entry) => entry.id).join(', ')
      throw new KvCacheInputError(
        `Unknown ${name}: "${String(dtype)}". Pick one of ${known}.`,
      )
    }
  }
  if (!Number.isFinite(contextLength) || contextLength < 1) {
    throw new KvCacheInputError('Context length must be a whole number of at least 1.')
  }
  if (!Number.isInteger(contextLength)) {
    throw new KvCacheInputError('Context length must be a whole number of tokens.')
  }
  if (!Number.isFinite(sequenceCount) || sequenceCount < 1) {
    throw new KvCacheInputError('Sequence count must be a whole number of at least 1.')
  }
  if (!Number.isInteger(sequenceCount)) {
    throw new KvCacheInputError('Sequence count must be a whole number.')
  }
  if (contextLength > 100_000_000) {
    throw new KvCacheInputError('Context length is larger than this calculator handles.')
  }
  if (sequenceCount > 100_000) {
    throw new KvCacheInputError('Sequence count is larger than this calculator handles.')
  }
}

export function computeKvCache(
  raw: RawConfig,
  rawOptions: Partial<ComputeOptions> = {},
): ComputeResult {
  const options: ComputeOptions = {
    contextLength: rawOptions.contextLength ?? 32768,
    sequenceCount: rawOptions.sequenceCount ?? 1,
    kvCacheDtype: rawOptions.kvCacheDtype ?? DEFAULT_DTYPE,
    indexerDtype: rawOptions.indexerDtype ?? DEFAULT_DTYPE,
  }

  validate(options)

  const { inner, outer } = unwrapConfig(raw)
  const detection = detectArchitecture(inner, outer)
  const kvBytes = dtypeBytes(options.kvCacheDtype)
  const indexerBytes = dtypeBytes(options.indexerDtype)

  let computation: FamilyComputation

  if (detection.family === 'dsv4') {
    computation = computeDsv4(inner, options, kvBytes, indexerBytes)
  } else if (detection.family === 'hybrid_linear') {
    computation = computeHybrid(inner, options, kvBytes)
  } else if (detection.family === 'mla') {
    computation = computeMla(inner, options, kvBytes, indexerBytes)
  } else if (detection.family === 'gqa') {
    computation = computeGqa(inner, options, kvBytes)
  } else {
    // Best effort: reuse whichever shape the config supports.
    if (readNumber(inner, 'kv_lora_rank') !== undefined) {
      computation = computeMla(inner, options, kvBytes, indexerBytes)
      computation.bestEffort = true
      computation.bestEffortReason =
        'model_type was not recognised. The MLA shape was assumed from kv_lora_rank, so treat this as a best effort estimate.'
    } else if (readNumber(inner, 'num_key_value_heads') !== undefined) {
      computation = computeGqa(inner, options, kvBytes)
      computation.bestEffort = true
      computation.bestEffortReason =
        'model_type was not recognised. The grouped query attention shape was assumed from num_key_value_heads, so treat this as a best effort estimate.'
    } else {
      throw new KvCacheInputError(
        `This config has no fields this calculator recognises, so the cache size cannot be derived. model_type is "${detection.modelType}".`,
      )
    }
  }

  const contextLength = options.contextLength
  const sequenceCount = options.sequenceCount

  const components: ComponentBreakdown[] = []

  const attentionBytesPerSequence = sumKinds(computation.attention.kinds)
  components.push({
    id: 'attention',
    label: 'Attention cache',
    formula: computation.attention.kinds.map((kind) => `${kind.layers} x ${kind.label}`).join(' + '),
    bytesPerToken: attentionBytesPerSequence / contextLength,
    bytesPerSequence: attentionBytesPerSequence,
    totalBytes: attentionBytesPerSequence * sequenceCount,
    inputs: computation.attention.kinds.flatMap((kind) => [
      { key: kind.label, value: `${kind.layers} layers x ${kind.bytesPerEntry} bytes per entry` },
    ]),
    note: computation.attention.note,
  })

  if (computation.indexer) {
    const indexerBytesPerSequence = computation.indexer.layers * computation.indexer.bytesPerEntry * computation.indexer.entriesPerSequence
    components.push({
      id: 'indexer',
      label: 'Indexer key cache',
      formula: `${computation.indexer.layers} layers x ${computation.indexer.formula}`,
      bytesPerToken: indexerBytesPerSequence / contextLength,
      bytesPerSequence: indexerBytesPerSequence,
      totalBytes: indexerBytesPerSequence * sequenceCount,
      inputs: [{ key: 'indexer layers', value: computation.indexer.layers }],
      note: computation.indexer.note,
    })
  }

  if (computation.state) {
    const stateBytesPerSequence = computation.state.bytesPerSequence
    components.push({
      id: 'state',
      label: 'Linear attention state',
      formula: computation.state.formula,
      bytesPerToken: stateBytesPerSequence / contextLength,
      bytesPerSequence: stateBytesPerSequence,
      totalBytes: stateBytesPerSequence * sequenceCount,
      inputs: computation.state.inputs,
      note: computation.state.note,
    })
  }

  const totalBytes = components.reduce((total, component) => total + component.totalBytes, 0)
  const bytesPerToken = totalBytes / (contextLength * sequenceCount)

  const assumptions = [...computation.assumptions]
  assumptions.push(
    'This is the logical size of the cache. Engines add their own overhead, for example FP8 block scale factors. vLLM stores 656 bytes per token per layer for DeepSeek-V3.2 where this formula gives 576, the difference being the scale factors.',
  )

  const maxPositionEmbeddings =
    readNumber(inner, 'max_position_embeddings') ?? readNumber(outer, 'max_position_embeddings') ?? null

  if (maxPositionEmbeddings && contextLength > maxPositionEmbeddings) {
    assumptions.push(
      `Context length ${contextLength} is above this model's max_position_embeddings of ${maxPositionEmbeddings}. The number below assumes the model would actually accept it.`,
    )
  }

  const steps: Array<{ label: string; detail: string }> = [
    {
      label: 'Read the config',
      detail: `model_type is ${detection.modelType}. ${detection.reason}`,
    },
    {
      label: 'Pick the formula',
      detail: `${detection.label}. ${computation.layerSplit.total} layers: ${describeSplit(computation.layerSplit)}.`,
    },
    {
      label: 'Size one entry per layer',
      detail: computation.attention.kinds.map((kind) => `${kind.label}: ${kind.formula}`).join('; '),
    },
    {
      label: 'Scale by tokens',
      detail: `Each layer holds ${contextLength} entries per sequence at this context length.`,
    },
    {
      label: 'Multiply by sequences',
      detail: `${sequenceCount} sequence${sequenceCount === 1 ? '' : 's'}.`,
    },
  ]

  if (computation.state) {
    steps.push({
      label: 'Add the constant state',
      detail:
        'Linear attention layers hold a fixed size state that does not grow with context, so it is added once per sequence.',
    })
  }

  const dtypeSupportList: DtypeSupport[] = dtypeSupport(detection.family, 'kv')
  const indexerSupport = computation.indexer
    ? dtypeSupport(detection.family, 'indexer')
    : null

  if (detection.family === 'unknown') {
    assumptions.push('The architecture was not recognised, so this is a best effort estimate.')
  }

  const isBestEffort = detection.bestEffort || computation.bestEffort
  if (isBestEffort && computation.bestEffortReason) {
    assumptions.push(computation.bestEffortReason)
  }

  return {
    totalBytes,
    bytesPerToken,
    architecture: {
      family: computation.family,
      modelType: detection.modelType,
      label: detection.label,
    },
    layerSplit: computation.layerSplit,
    components,
    constants: computation.constants,
    steps,
    assumptions,
    bestEffort: detection.bestEffort || computation.bestEffort,    dtypeSupport: dtypeSupportList,
    indexerDtypeSupport: indexerSupport,
    contextLength,
    sequenceCount,
    maxPositionEmbeddings,
  }
}

function describeSplit(split: LayerSplit): string {
  const parts: string[] = []
  if (split.fullAttention) parts.push(`${split.fullAttention} full attention`)
  if (split.slidingAttention) parts.push(`${split.slidingAttention} sliding window`)
  if (split.linearAttention) parts.push(`${split.linearAttention} linear attention`)
  if (split.compressed) {
    for (const entry of split.compressed) {
      parts.push(`${entry.layers} at ratio ${entry.ratio}`)
    }
  }
  return parts.length > 0 ? parts.join(', ') : 'no per layer detail'
}
