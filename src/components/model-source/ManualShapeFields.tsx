import NumberField from '@/components/ui/number-field'
import SegmentedControl from '@/components/ui/segmented'
import { TIE_OPTIONS } from '@/components/model-source/labels'
import type { ConfigSourceInputs } from '@/lib/use-config-source'

/**
 * The manual model fields, shared by every calculator that can describe a model
 * by hand.
 *
 * The labels are the config keys themselves, so a reader can copy them straight
 * out of a config.json they already have.
 */

interface ManualShapeFieldsProps {
  /** Prefix for every field id, so two forms on one page cannot collide. */
  idPrefix: string
  inputs: ConfigSourceInputs
  update: (patch: Partial<ConfigSourceInputs>) => void
}

export default function ManualShapeFields({
  idPrefix,
  inputs,
  update,
}: ManualShapeFieldsProps) {
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2">
        <NumberField
          id={`${idPrefix}-hidden`}
          label="hidden_size"
          min={1}
          value={inputs.hiddenSize}
          onChange={(hiddenSize) => update({ hiddenSize })}
        />
        <NumberField
          id={`${idPrefix}-ffn`}
          label="intermediate_size"
          hint="The width of the dense feed forward block."
          min={1}
          value={inputs.intermediateSize}
          onChange={(intermediateSize) => update({ intermediateSize })}
        />
        <NumberField
          id={`${idPrefix}-layers`}
          label="num_hidden_layers"
          min={1}
          value={inputs.numLayers}
          onChange={(numLayers) => update({ numLayers })}
        />
        <NumberField
          id={`${idPrefix}-vocab`}
          label="vocab_size"
          hint="This value sets the size of the embedding table."
          min={1}
          value={inputs.vocabSize}
          onChange={(vocabSize) => update({ vocabSize })}
        />
        <NumberField
          id={`${idPrefix}-heads`}
          label="num_attention_heads"
          min={1}
          value={inputs.attentionHeads}
          onChange={(attentionHeads) => update({ attentionHeads })}
        />
        <NumberField
          id={`${idPrefix}-kv-heads`}
          label="num_key_value_heads"
          hint="Fewer than the attention heads means grouped query attention, so the cache is smaller."
          min={1}
          value={inputs.kvHeads}
          onChange={(kvHeads) => update({ kvHeads })}
        />
        <NumberField
          id={`${idPrefix}-head-dim`}
          label="head_dim"
          min={1}
          value={inputs.headDim}
          onChange={(headDim) => update({ headDim })}
        />
        <NumberField
          id={`${idPrefix}-experts`}
          label="routed experts"
          hint="Use 0 for a dense model."
          min={0}
          value={inputs.routedExperts}
          onChange={(routedExperts) => update({ routedExperts })}
        />
        <NumberField
          id={`${idPrefix}-topk`}
          label="experts per token"
          hint="The router top-k. This value sets the weights one token reads."
          min={0}
          value={inputs.expertsPerToken}
          onChange={(expertsPerToken) => update({ expertsPerToken })}
        />
        <NumberField
          id={`${idPrefix}-expert-ffn`}
          label="moe_intermediate_size"
          hint="The width of one expert."
          min={0}
          value={inputs.moeIntermediateSize}
          onChange={(moeIntermediateSize) => update({ moeIntermediateSize })}
        />
        <NumberField
          id={`${idPrefix}-moe-layers`}
          label="layers with experts"
          min={0}
          value={inputs.moeLayers}
          onChange={(moeLayers) => update({ moeLayers })}
        />
      </div>

      <SegmentedControl
        legend="Embedding and head"
        options={TIE_OPTIONS}
        value={inputs.tieEmbeddings}
        onValueChange={(tieEmbeddings) => update({ tieEmbeddings })}
        wrap
      />
    </>
  )
}
