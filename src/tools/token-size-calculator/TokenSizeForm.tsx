import {
  ModelIdField,
  ProviderPicker,
  TokenField,
} from '@/components/model-source/ModelSourceFields'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  formatTokenAmountShort,
  parseTokenAmount,
  TOKENIZER_FAMILY_LABELS,
  TOKENIZER_PRESETS,
  type TokenizerError,
  type TokenizerProfile,
} from '@/lib/token-size'
import type { TokenizerStatus } from '@/lib/use-tokenizer'

import type { TokenSizeInputs } from './useTokenSizeState'

/** One-tap token amounts, chosen so a range of answers shows up. */
const AMOUNT_PRESETS = [1_000, 128_000, 1_000_000, 1_000_000_000, 1_000_000_000_000]

interface TokenSizeFormProps {
  inputs: TokenSizeInputs
  update: (patch: Partial<TokenSizeInputs>) => void
  status: TokenizerStatus
  profile: TokenizerProfile | null
  error: TokenizerError | null
}

export default function TokenSizeForm({
  inputs,
  update,
  status,
  profile,
  error,
}: TokenSizeFormProps) {
  const tokens = parseTokenAmount(inputs.tokens)
  const invalid = tokens === null

  return (
    <form className="flex min-w-0 flex-col gap-6" onSubmit={(event) => event.preventDefault()}>
      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Model</legend>

        <ProviderPicker
          value={inputs.provider}
          onValueChange={(provider) => update({ provider })}
        />

        <ModelIdField
          id="token-size-model-id"
          listId="token-size-model-presets"
          value={inputs.modelId}
          onChange={(modelId) => update({ modelId })}
          status={status}
          presets={TOKENIZER_PRESETS}
          summary={
            profile && (
              <>
                {TOKENIZER_FAMILY_LABELS[profile.family]} ·{' '}
                {profile.vocabSize.toLocaleString('en-US')} token vocab
                {profile.estimated && ' · estimate only'}
              </>
            )
          }
          idleHint="Type a model id, or pick a suggestion whose tokenizer is public."
          loadingHint="Reading the tokenizer"
          errorHint={error?.message ?? 'Could not read that tokenizer.'}
        />

        <TokenField
          id="token-size-token"
          label={inputs.provider === 'huggingface' ? 'hf_token' : 'ms_token'}
          value={inputs.token}
          onChange={(token) => update({ token })}
          hint="Only needed for gated or private repos. Sent to the hub and nowhere else."
        />
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm font-medium">Token count</legend>

        <div className="flex flex-wrap gap-2">
          {AMOUNT_PRESETS.map((amount) => {
            const active = tokens === amount
            return (
              <Button
                key={amount}
                type="button"
                size="sm"
                variant={active ? 'secondary' : 'outline'}
                aria-pressed={active}
                onClick={() => update({ tokens: String(amount) })}
              >
                {formatTokenAmountShort(amount)}
              </Button>
            )
          })}
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="token-size-tokens" className="text-sm font-medium">
            Amount of tokens
          </label>
          <Input
            id="token-size-tokens"
            inputMode="decimal"
            autoComplete="off"
            spellCheck={false}
            placeholder="e.g. 128k, 1.5M, 1B"
            value={inputs.tokens}
            onChange={(event) => update({ tokens: event.target.value })}
            aria-invalid={invalid}
            aria-describedby={invalid ? 'token-size-tokens-error' : undefined}
          />
          {invalid ? (
            <p id="token-size-tokens-error" className="text-xs text-rose-600 dark:text-rose-400">
              Enter a whole number of tokens, or a number with a k, m, b, or t suffix.
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              The number of tokens to price in bytes, for the model above.
            </p>
          )}
        </div>
      </fieldset>
    </form>
  )
}