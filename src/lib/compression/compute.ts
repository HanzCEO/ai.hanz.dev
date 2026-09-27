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
export function keptCostAt(ratePerMillion: number, tokens: number): number {
  return (ratePerMillion * tokens) / 1_000_000
}

/**
 * Dollars for the one request that carries the summary.
 *
 * The summary is capped at a share of the model window, so its size does not
 * move with the session. The session argument is here so a caller can see that
 * the answer does not depend on it.
 */
export function summaryCostAt(
  result: Pick<CompressionResult, 'summaryUnitPerMillion' | 'summaryTokens'>,
  _sessionTokens?: number,
): number {
  return keptCostAt(result.summaryUnitPerMillion, result.summaryTokens)
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
 * Finds the session size at which a summary starts to pay.
 *
 * The whole model is one line of algebra. A kept session of X tokens bills X
 * times a blended rate, where the blend is the miss price and the cache price
 * weighted by the mix, plus the output the reply writes. A summary is capped at
 * A percent of the model window, so it bills a fixed number of tokens at the
 * full input price, because no cache holds new text. The kept cost climbs with
 * the session and the summary cost does not, so the two cross at one size.
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

  // Dollars for 1M session tokens, with the context split across miss and cache.
  const missPart = (missPercent * inputs.inputPrice) / inputShare
  const cachePart = (cachePercent * inputs.cachedInputPrice) / inputShare
  const keptOutputPart = (outputPercent * inputs.outputPrice) / inputShare
  const keptRatePerMillion = missPart + cachePart + keptOutputPart

  // Dollars for 1M summary tokens. A summary is new text, so every token of it
  // pays the full input price, and it writes a reply at the same ratio.
  const summaryInputPart = inputs.inputPrice
  const summaryOutputPart = outputPerInput * inputs.outputPrice
  const summaryUnitPerMillion = summaryInputPart + summaryOutputPart

  const summaryTokens = (inputs.compressionPercent / 100) * CONTEXT_WINDOW_TOKENS
  const summaryCost = keptCostAt(summaryUnitPerMillion, summaryTokens)

  // Where the climbing kept line meets the flat summary line. The summary costs
  // the same at any session size, so the crossing is that fixed cost divided by
  // the kept rate per token.
  const breakEvenSessionTokens = (summaryCost * 1_000_000) / keptRatePerMillion

  // The share at which the crossing point lands exactly on the full window, so
  // a summary capped above it never pays off.
  const breakEvenPercent = (100 * keptRatePerMillion) / summaryUnitPerMillion
  const compressingWins = breakEvenSessionTokens < CONTEXT_WINDOW_TOKENS

  const keptAtWindow = keptCostAt(keptRatePerMillion, CONTEXT_WINDOW_TOKENS)
  const summaryAtWindow = summaryCostAt({ summaryUnitPerMillion, summaryTokens })
  const savingAtWindow = keptAtWindow - summaryAtWindow

  const steps: CompressionStep[] = [
    {
      label: 'Output per input token',
      detail: `Your mix writes ${outputPercent} output tokens for every ${inputShare} input tokens, so the reply is ${outputPerInput.toFixed(4)} times the context.`,
    },
    {
      label: 'Rate for a kept session',
      detail: `${missPercent} percent of the context at $${inputs.inputPrice}, ${cachePercent} percent at $${inputs.cachedInputPrice}, and the reply at $${inputs.outputPrice} come to $${keptRatePerMillion.toFixed(4)} for 1M session tokens.`,
    },
    {
      label: 'Rate for summary tokens',
      detail: `Every summary token is new text at the full input price, and the reply adds ${outputPerInput.toFixed(4)} times $${inputs.outputPrice}, which is $${summaryUnitPerMillion.toFixed(4)} for 1M summary tokens.`,
    },
    {
      label: 'Size of the summary',
      detail: `A ${inputs.compressionPercent} percent cap on a ${CONTEXT_WINDOW_TOKENS.toLocaleString('en-US')} token window is ${summaryTokens.toLocaleString('en-US')} tokens, whatever the length of the session.`,
    },
    {
      label: 'Session size where the two agree',
      detail: compressingWins
        ? `Carrying the summary costs $${summaryCost.toFixed(4)} at any size, so it pays once the session passes ${Math.round(breakEvenSessionTokens).toLocaleString('en-US')} tokens.`
        : `Carrying the summary costs $${summaryCost.toFixed(4)} at any size, which only matches the kept session at ${Math.round(breakEvenSessionTokens).toLocaleString('en-US')} tokens, past the ${CONTEXT_WINDOW_TOKENS.toLocaleString('en-US')} token window.`,
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
      key: 'summary cap',
      value: inputs.compressionPercent,
      source: 'Percent of the model window the summary is capped at.',
    },
    {
      key: 'context window',
      value: CONTEXT_WINDOW_TOKENS,
      source: 'The largest context a current model serves.',
    },
    {
      key: 'summary size',
      value: summaryTokens,
      source: 'Tokens the summary is capped at, which is the same at any session length.',
    },
  ]

  const assumptions = [
    'The summary cap is a share of the model window, so the summary is the same size whatever the session length.',
    'The context is input tokens, and the reply is the only output the request bills.',
    'The reply scales with the context, so a shorter context writes a shorter reply.',
    'A summary is new text, so every token of it is billed at the full input price.',
    'The context is already in the cache, so the kept path pays the cached price on its cache share.',
    'The call that writes the summary is not billed here, because it is a separate request.',
  ]

  return {
    keptRatePerMillion,
    summaryUnitPerMillion,
    summaryTokens,
    summaryCost,
    breakEvenSessionTokens,
    breakEvenPercent,
    compressingWins,
    keptAtWindow,
    summaryAtWindow,
    savingAtWindow,
    outputPerInput,
    keptParts: { miss: missPart, cache: cachePart, output: keptOutputPart },
    summaryParts: { input: summaryInputPart, output: summaryOutputPart },
    normalized,
    compressionPercent: inputs.compressionPercent,
    contextWindowTokens: CONTEXT_WINDOW_TOKENS,
    steps,
    constants,
    assumptions,
  }
}
