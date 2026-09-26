import {
  CONTEXT_WINDOW_TOKENS,
  MAX_COMPRESSION_PERCENT,
  MIN_COMPRESSION_PERCENT,
} from './presets'
import {
  CompressionInputError,
  type CompressionConstant,
  type CompressionInputField,
  type CompressionInputs,
  type CompressionResult,
  type CompressionStep,
} from './types'

/** Dollars for a request of this many tokens, at a rate given for 1M tokens. */
export function requestCost(ratePerMillion: number, tokens: number): number {
  return (ratePerMillion * tokens) / 1_000_000
}

function requireFinite(
  value: number,
  field: CompressionInputField,
  label: string,
): void {
  if (!Number.isFinite(value)) {
    throw new CompressionInputError(`${label} must be a number.`, field)
  }
}

function validate(inputs: CompressionInputs): void {
  requireFinite(inputs.inputPrice, 'inputPrice', 'The input price')
  if (inputs.inputPrice <= 0) {
    throw new CompressionInputError(
      'The input price must be more than zero.',
      'inputPrice',
    )
  }

  requireFinite(inputs.cachedInputPrice, 'cachedInputPrice', 'The cached input price')
  if (inputs.cachedInputPrice < 0) {
    throw new CompressionInputError(
      'The cached input price cannot be negative.',
      'cachedInputPrice',
    )
  }

  requireFinite(inputs.outputPrice, 'outputPrice', 'The output price')
  if (inputs.outputPrice < 0) {
    throw new CompressionInputError('The output price cannot be negative.', 'outputPrice')
  }

  requireFinite(inputs.mix.missPercent, 'missPercent', 'The miss share')
  if (inputs.mix.missPercent < 0) {
    throw new CompressionInputError('The miss share cannot be negative.', 'missPercent')
  }

  requireFinite(inputs.mix.cachePercent, 'cachePercent', 'The cache share')
  if (inputs.mix.cachePercent < 0) {
    throw new CompressionInputError('The cache share cannot be negative.', 'cachePercent')
  }

  requireFinite(inputs.mix.outputPercent, 'outputPercent', 'The output share')
  if (inputs.mix.outputPercent < 0) {
    throw new CompressionInputError('The output share cannot be negative.', 'outputPercent')
  }

  if (inputs.mix.missPercent + inputs.mix.cachePercent <= 0) {
    throw new CompressionInputError(
      'The miss share and the cache share cannot both be zero, because the context is input tokens.',
      null,
    )
  }

  requireFinite(inputs.compressionPercent, 'compressionPercent', 'The compression share')
  if (
    inputs.compressionPercent < MIN_COMPRESSION_PERCENT ||
    inputs.compressionPercent > MAX_COMPRESSION_PERCENT
  ) {
    throw new CompressionInputError(
      `The compression share must be between ${MIN_COMPRESSION_PERCENT} and ${MAX_COMPRESSION_PERCENT}.`,
      'compressionPercent',
    )
  }
}

/**
 * Costs the next request against a session two ways, and finds the compression
 * share at which the two agree.
 *
 * The whole model is one line of algebra. A kept session of X tokens bills
 * X times a blended rate, where the blend is the miss price and the cache price
 * weighted by the mix, plus the output the reply writes. A summarised session
 * bills A percent of X at the full input price, because a summary is new text
 * that no cache holds. Both costs are linear in X with no constant term, so
 * they meet only at zero, and the winner is decided by the two slopes alone.
 */
export function estimateCompression(inputs: CompressionInputs): CompressionResult {
  validate(inputs)

  const { missPercent, cachePercent, outputPercent } = inputs.mix
  const inputShare = missPercent + cachePercent
  const totalShare = inputShare + outputPercent

  // The shares are shown back to the reader as they were entered, scaled to sum
  // to 100 so a mix that does not add up cannot quietly change the blend.
  const normalized = {
    missPercent: (missPercent / totalShare) * 100,
    cachePercent: (cachePercent / totalShare) * 100,
    outputPercent: (outputPercent / totalShare) * 100,
  }

  const outputPerInput = outputPercent / inputShare

  // Dollars for 1M context tokens, with the context split across miss and cache.
  const missPart = (missPercent * inputs.inputPrice) / inputShare
  const cachePart = (cachePercent * inputs.cachedInputPrice) / inputShare
  const keptOutputPart = (outputPercent * inputs.outputPrice) / inputShare
  const neverRatePerMillion = missPart + cachePart + keptOutputPart

  const compressShare = inputs.compressionPercent / 100
  const compressedInputPart = compressShare * inputs.inputPrice
  const compressedOutputPart = compressShare * outputPerInput * inputs.outputPrice
  const compressRatePerMillion = compressedInputPart + compressedOutputPart

  // The share where the summarised line crosses the kept line. The summarised
  // line always pays the full input price, so this depends on the input price
  // and never on the cached one alone.
  const breakEvenPercent =
    (100 * neverRatePerMillion) /
    (inputs.inputPrice + outputPerInput * inputs.outputPrice)

  const neverAtWindow = requestCost(neverRatePerMillion, CONTEXT_WINDOW_TOKENS)
  const compressAtWindow = requestCost(compressRatePerMillion, CONTEXT_WINDOW_TOKENS)
  const savingAtWindow = neverAtWindow - compressAtWindow

  const steps: CompressionStep[] = [
    {
      label: 'Output per input token',
      detail: `Your mix writes ${outputPercent} output tokens for every ${inputShare} input tokens, so the reply is ${outputPerInput.toFixed(4)} times the context.`,
    },
    {
      label: 'Blended rate for a kept session',
      detail: `${missPercent} percent of the context at $${inputs.inputPrice}, ${cachePercent} percent at $${inputs.cachedInputPrice}, and the reply at $${inputs.outputPrice} come to $${neverRatePerMillion.toFixed(4)} for 1M context tokens.`,
    },
    {
      label: 'Rate for a summarised session',
      detail: `Keeping ${inputs.compressionPercent} percent of the context bills $${compressedInputPart.toFixed(4)} of input and $${compressedOutputPart.toFixed(4)} of output, which is $${compressRatePerMillion.toFixed(4)} for 1M context tokens.`,
    },
    {
      label: 'Break-even share',
      detail: `The two rates agree when the summary keeps ${breakEvenPercent.toFixed(2)} percent of the context.`,
    },
    {
      label: 'Cost at the full window',
      detail: `At 1M context tokens the kept session costs $${neverAtWindow.toFixed(4)} and the summarised one costs $${compressAtWindow.toFixed(4)}.`,
    },
  ]

  const constants: CompressionConstant[] = [
    {
      key: 'input price',
      value: inputs.inputPrice,
      source: 'Dollars for 1M input tokens that miss the cache.',
    },
    {
      key: 'cached input price',
      value: inputs.cachedInputPrice,
      source: 'Dollars for 1M input tokens that hit the cache.',
    },
    {
      key: 'output price',
      value: inputs.outputPrice,
      source: 'Dollars for 1M output tokens.',
    },
    {
      key: 'miss share',
      value: missPercent,
      source: 'Percent of the billed tokens that miss the cache.',
    },
    {
      key: 'cache share',
      value: cachePercent,
      source: 'Percent of the billed tokens that hit the cache.',
    },
    {
      key: 'output share',
      value: outputPercent,
      source: 'Percent of the billed tokens the reply writes.',
    },
    {
      key: 'context kept',
      value: inputs.compressionPercent,
      source: 'Percent of the session a summary keeps.',
    },
    {
      key: 'context window',
      value: CONTEXT_WINDOW_TOKENS,
      source: 'The largest context a current model serves.',
    },
  ]

  const assumptions = [
    'The context is input tokens, and the reply is the only output the request bills.',
    'The reply scales with the context, so a shorter context writes a shorter reply.',
    'A summary is new text, so every token it keeps is billed at the full input price.',
    'The context is already in the cache, so the kept path pays the cached price on its cache share.',
    'The call that writes the summary is not billed here, because it is a separate request.',
  ]

  return {
    neverRatePerMillion,
    compressRatePerMillion,
    breakEvenPercent,
    compressingWins: compressRatePerMillion < neverRatePerMillion,
    neverAtWindow,
    compressAtWindow,
    savingAtWindow,
    outputPerInput,
    neverParts: { miss: missPart, cache: cachePart, output: keptOutputPart },
    compressParts: { input: compressedInputPart, output: compressedOutputPart },
    normalized,
    compressionPercent: inputs.compressionPercent,
    contextWindowTokens: CONTEXT_WINDOW_TOKENS,
    steps,
    constants,
    assumptions,
  }
}
