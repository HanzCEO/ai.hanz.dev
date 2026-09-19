import type { RawConfig } from './types'

/** Raised when pasted text cannot be read as a model config. */
export class ConfigParseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ConfigParseError'
  }
}

/**
 * Turns pasted text into a config object.
 *
 * This exists so a user who already downloaded a config.json, or who runs a
 * private model that is not on a hub, can still size it. Every failure carries
 * a message worth showing, since the input came from outside the app.
 */
export function parseConfigText(text: string): RawConfig {
  const trimmed = text.trim()
  if (trimmed === '') {
    throw new ConfigParseError('Paste a config.json, or switch to the model id input.')
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    throw new ConfigParseError('That is not valid JSON. Paste the whole config.json file.')
  }

  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new ConfigParseError(
      'A model config is a JSON object. This looks like an array or a single value.',
    )
  }

  const config = parsed as RawConfig
  if (Object.keys(config).length === 0) {
    throw new ConfigParseError('That config object is empty, so it carries no shape to read.')
  }

  return config
}
