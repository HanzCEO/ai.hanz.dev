import { Eye, EyeOff, Loader2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import { Input } from '@/components/ui/input'
import SegmentedControl, { type SegmentedOption } from '@/components/ui/segmented'
import { PROVIDER_LIST, PROVIDER_ORDER, type Provider } from '@/lib/kvcache'

/**
 * The model id and provider fields, which every calculator that reads a config
 * from a hub needs.
 *
 * The fields are identical in structure across tools and differ only in their
 * copy, so the structure lives here and the copy is passed in. The status line
 * is the interesting part: it has to say, in one short line, whether the config
 * is being fetched, what was found, or why it could not be read.
 */

/** What the field is currently able to say about the config. */
export type ModelIdStatus = 'idle' | 'loading' | 'ready' | 'error' | 'not-moe'

/** A suggestion offered through the datalist. */
export interface ModelPreset {
  id: string
  note: string
}

/**
 * The providers a config can be read from, in the order they are offered. Built
 * once here so no page has to know how a provider turns into a button label.
 */
const PROVIDER_OPTIONS: SegmentedOption<Provider>[] = PROVIDER_ORDER.map((provider) => ({
  id: provider,
  label: PROVIDER_LIST.find((spec) => spec.id === provider)?.label ?? provider,
}))

interface ProviderPickerProps {
  value: Provider
  onValueChange: (value: Provider) => void
}

export function ProviderPicker({ value, onValueChange }: ProviderPickerProps) {
  return (
    <SegmentedControl
      legend="Provider"
      options={PROVIDER_OPTIONS}
      value={value}
      onValueChange={onValueChange}
    />
  )
}

interface ModelIdFieldProps {
  id: string
  value: string
  onChange: (value: string) => void
  status: ModelIdStatus
  /** Rendered when the status is ready, for example the detected architecture. */
  summary?: ReactNode
  label?: string
  /** Copy for the idle status. */
  idleHint?: string
  /** Copy for the error status. */
  errorHint?: string
  /** Copy for the not-moe status, for tools that only apply to some models. */
  notMoeHint?: string
  /** Suggestions rendered into a datalist, which also enables autocomplete. */
  presets?: readonly ModelPreset[]
  /** Id for the datalist, when a page already has one it wants to keep. */
  listId?: string
  placeholder?: string
}

export function ModelIdField({
  id,
  value,
  onChange,
  status,
  summary,
  label = 'Model id',
  idleHint = 'Type a model id, or pick a suggestion.',
  errorHint = 'Could not read that config.',
  notMoeHint,
  presets,
  listId,
  placeholder = 'owner/name',
}: ModelIdFieldProps) {
  const presetsId = listId ?? `${id}-presets`

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <Input
        id={id}
        list={presets ? presetsId : undefined}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete="off"
      />
      {presets && (
        <datalist id={presetsId}>
          {presets.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.note}
            </option>
          ))}
        </datalist>
      )}
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        {status === 'loading' && (
          <>
            <Loader2 className="size-3 animate-spin" aria-hidden="true" />
            Reading the config
          </>
        )}
        {status === 'idle' && idleHint}
        {status === 'ready' && summary}
        {status === 'not-moe' && notMoeHint}
        {status === 'error' && errorHint}
      </p>
    </div>
  )
}

/**
 * Whether the token field is showing its value in clear text. Kept private: the
 * field owns the state, and nothing else needs to read it.
 */
function useTokenVisibility() {
  const [visible, setVisible] = useState(false)
  return {
    visible,
    toggle: () => setVisible((current) => !current),
  }
}

interface TokenFieldProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  hint?: string
}

/**
 * A token field that hides what was typed and never leaves the page. The
 * reveal button is the only way to check a paste, so it is part of the field
 * rather than something each form adds for itself.
 */
export function TokenField({ id, label, value, onChange, hint }: TokenFieldProps) {
  const { visible, toggle } = useTokenVisibility()

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <div className="relative">
        <Input
          id={id}
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Optional"
          spellCheck={false}
          autoComplete="off"
          className="pr-9"
        />
        <button
          type="button"
          onClick={toggle}
          aria-label={visible ? 'Hide the token' : 'Show the token'}
          className="text-muted-foreground hover:text-foreground absolute inset-y-0 right-0 flex w-9 items-center justify-center"
        >
          {visible ? (
            <EyeOff className="size-4" aria-hidden="true" />
          ) : (
            <Eye className="size-4" aria-hidden="true" />
          )}
        </button>
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

interface ConfigPasteFieldProps {
  id: string
  value: string
  onChange: (value: string) => void
  placeholder: string
}

/**
 * A textarea for a pasted config.json.
 *
 * Nothing is sent anywhere: the config is parsed in the browser, which the hint
 * says out loud because pasting a private model's config is otherwise a
 * reasonable thing to hesitate over.
 */
export function ConfigPasteField({ id, value, onChange, placeholder }: ConfigPasteFieldProps) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        config.json
      </label>
      <textarea
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={false}
        placeholder={placeholder}
        className="border-input focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 min-h-40 w-full rounded-lg border bg-transparent px-2.5 py-2 font-mono text-xs outline-none focus-visible:ring-3"
      />
      <p className="text-xs text-muted-foreground">
        Nothing is uploaded. The config is read in the browser.
      </p>
    </div>
  )
}
