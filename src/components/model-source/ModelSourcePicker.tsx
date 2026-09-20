import { type ReactNode } from 'react'

import {
  ConfigPasteField,
  ModelIdField,
  ProviderPicker,
  TokenField,
  type ModelIdStatus,
  type ModelPreset,
} from '@/components/model-source/ModelSourceFields'
import ManualShapeFields from '@/components/model-source/ManualShapeFields'
import { MODE_LABELS } from '@/components/model-source/labels'
import SegmentedControl from '@/components/ui/segmented'
import { PROVIDER_LIST } from '@/lib/kvcache'
import type { ConfigSourceInputs } from '@/lib/use-config-source'

/**
 * The model source switch, and the fields that belong to each source.
 *
 * Every calculator that needs a model shape offers the same three ways in, so
 * the switch and the three field sets live here rather than in each form. A
 * page passes its own status line and its own suggestion list.
 */

interface ModelSourcePickerProps {
  /** Prefix for every field id, so two forms on one page cannot collide. */
  idPrefix: string
  inputs: ConfigSourceInputs
  update: (patch: Partial<ConfigSourceInputs>) => void
  /** What the resolved config is currently able to say. */
  status: ModelIdStatus
  /** Rendered under the model id field once the config is ready. */
  summary?: ReactNode
  /** Suggestions rendered into the model id datalist. */
  presets?: readonly ModelPreset[]
  /** Copy for the idle status on the model id field. */
  idleHint?: string
  /** Copy for the error status on the model id field. */
  errorHint?: string
}

export default function ModelSourcePicker({
  idPrefix,
  inputs,
  update,
  status,
  summary,
  presets,
  idleHint,
  errorHint,
}: ModelSourcePickerProps) {
  const providerSpec = PROVIDER_LIST.find((spec) => spec.id === inputs.provider)

  return (
    <>
      <SegmentedControl
        legend="Model source"
        options={MODE_LABELS}
        value={inputs.mode}
        onValueChange={(mode) => update({ mode })}
        wrap
      />

      {inputs.mode === 'hub' && (
        <>
          <ProviderPicker
            value={inputs.provider}
            onValueChange={(provider) => update({ provider })}
          />

          <ModelIdField
            id={`${idPrefix}-model-id`}
            listId={`${idPrefix}-model-presets`}
            value={inputs.modelId}
            onChange={(modelId) => update({ modelId })}
            status={status}
            presets={presets}
            summary={summary}
            idleHint={idleHint}
            errorHint={errorHint}
          />

          <TokenField
            id={`${idPrefix}-token`}
            label={providerSpec?.tokenLabel ?? 'token'}
            value={inputs.token}
            onChange={(token) => update({ token })}
            hint={providerSpec?.tokenHint}
          />
        </>
      )}

      {inputs.mode === 'paste' && (
        <ConfigPasteField
          id={`${idPrefix}-config-text`}
          value={inputs.configText}
          onChange={(configText) => update({ configText })}
          placeholder='{ "model_type": "qwen3", "num_hidden_layers": 36, ... }'
        />
      )}

      {inputs.mode === 'manual' && (
        <ManualShapeFields idPrefix={idPrefix} inputs={inputs} update={update} />
      )}
    </>
  )
}
