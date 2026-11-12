import FaqSection from '@/components/faq/FaqSection'
import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import { parseTokenAmount, TOKEN_SIZE_FAQ } from '@/lib/token-size'
import { useTokenizer } from '@/lib/use-tokenizer'
import TokenSizeForm from '@/tools/token-size-calculator/TokenSizeForm'
import TokenSizeResults from '@/tools/token-size-calculator/TokenSizeResults'
import { useTokenSizeState } from '@/tools/token-size-calculator/useTokenSizeState'

export default function TokenSizeCalculator() {
  const { inputs, update } = useTokenSizeState()
  const tokenizer = useTokenizer(inputs.provider, inputs.modelId, inputs.token)
  const tokensValue = parseTokenAmount(inputs.tokens)

  return (
    <div className="flex flex-col gap-10">
      <ToolBreadcrumb name="Token Size Calculator" />

      <ToolHeader
        title="How many GB is a token count?"
        description="Type a model's org/name and a token count, and the page sizes that many tokens in bytes, KB, MB, GB, TB, and PB — measured with the model's own tokenizer, read from HuggingFace or ModelScope."
        widthClass="max-w-2xl"
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-12">
        <TokenSizeForm
          inputs={inputs}
          update={update}
          status={tokenizer.status}
          profile={tokenizer.profile}
          error={tokenizer.error}
        />

        <TokenSizeResults
          status={tokenizer.status}
          profile={tokenizer.profile}
          url={tokenizer.url}
          error={tokenizer.error}
          tokensValue={tokensValue}
        />
      </div>

      <section
        aria-labelledby="token-size-source-heading"
        className="text-muted-foreground max-w-3xl text-sm"
      >
        <h2
          id="token-size-source-heading"
          className="text-foreground mb-2 text-lg font-medium tracking-tight"
        >
          Where the number comes from
        </h2>
        <p>
          Every model is measured against the same bundled sample: English prose, function
          definitions, numbers, symbols, URLs, and a paragraph of CJK text. The tokenizer.json is
          read from the hub in the browser, then run over that sample exactly as the model would
          split it — the same pretokenizer, the same merge ranks. Because the sample never changes,
          models on this page are compared on equal terms, and a bookmark of a size you computed
          can be reloaded and re-counted identically.
        </p>
      </section>

      <FaqSection
        id="token-size-faq-heading"
        heading="Questions about sizing tokens"
        items={TOKEN_SIZE_FAQ}
      />
    </div>
  )
}