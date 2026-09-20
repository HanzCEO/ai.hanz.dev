import type { ConfigSourceInputs, ConfigSourceMode } from '@/lib/use-config-source'

/**
 * The copy for the model source fields.
 *
 * Kept out of the component files so a plain module can hold the strings. That
 * lets a test read the offered set without rendering, and keeps every label in
 * one place for a copy review.
 */

/** The three ways a reader can describe a model. */
export const MODE_LABELS: Array<{ id: ConfigSourceMode; label: string; hint: string }> = [
  { id: 'hub', label: 'Model id', hint: 'Read the config from HuggingFace or ModelScope.' },
  { id: 'paste', label: 'Paste config', hint: 'Paste a config.json that you already have.' },
  { id: 'manual', label: 'Enter numbers', hint: 'Enter the model values yourself.' },
]

/** Whether the embedding and the language model head share one table. */
export const TIE_OPTIONS: Array<{
  id: ConfigSourceInputs['tieEmbeddings']
  label: string
  hint: string
}> = [
  {
    id: 'untied',
    label: 'Separate head',
    hint: 'The embedding and the language model head are 2 separate tables. Most models do this.',
  },
  {
    id: 'tied',
    label: 'Tied head',
    hint: 'The embedding and the language model head share 1 table, so the checkpoint is smaller by 1 embedding table.',
  },
]
