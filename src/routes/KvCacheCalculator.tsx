import { ToolBreadcrumb, ToolHeader } from '@/components/layout/ToolPage'
import { useConfigSource } from '@/lib/use-config-source'
import CalculatorForm from '@/tools/kv-cache-calculator/CalculatorForm'
import CalculatorResults from '@/tools/kv-cache-calculator/CalculatorResults'
import { useCalculatorState } from '@/tools/kv-cache-calculator/useCalculatorState'
import { useKvCacheLayer } from '@/tools/kv-cache-calculator/useKvCacheLayer'

export default function KvCacheCalculator() {
  const { inputs, update, seeded } = useCalculatorState()
  const configState = useConfigSource(inputs)
  const cache = useKvCacheLayer({ inputs, update, configState, seeded })

  return (
    <div className="flex flex-col gap-8">
      <ToolBreadcrumb name="KV Cache Calculator" />

      <ToolHeader
        title="KV Cache Calculator"
        description="Size the cache for a model, a context length, and a sequence count. The config can come from HuggingFace, from ModelScope, from a config.json you paste, or from values you enter yourself."
        widthClass="max-w-2xl"
      />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)] lg:gap-12">
        <CalculatorForm
          inputs={inputs}
          update={update}
          configState={configState}
          result={cache.result}
          contextError={cache.contextError}
          sequenceError={cache.sequenceError}
          onUseMaxContext={cache.onUseMaxContext}
        />

        <CalculatorResults
          result={cache.result}
          configState={configState}
          onSwitchProvider={cache.onSwitchProvider}
          computeError={cache.computeError}
        />
      </div>
    </div>
  )
}
