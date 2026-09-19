import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'

/**
 * Keeps a set of inputs in the query string so a calculation can be shared,
 * bookmarked, or reloaded.
 *
 * The URL is read only after mount, never during the first render. The
 * prerendered HTML was produced without any query string, so reading the URL
 * during render would make the first client render disagree with the server
 * markup and trip a hydration error.
 */

/** Describes how one field is parsed from, and written to, the query string. */
export interface UrlFieldSpec<T> {
  /** Query string parameter name. Ignored when omit is true. */
  param: string
  /** Value used when the URL does not supply one. */
  default: T
  /** Reads the raw parameter into a value, or null when it is not usable. */
  parse: (raw: string | null) => T | null
  /** Overrides how a value is written. Defaults to String(value). */
  serialize?: (value: T) => string
  /** When true the field is never read from or written to the URL. */
  omit?: boolean
}

/** A schema covering every key of the value shape. */
export type UrlSchema<T> = { [K in keyof T]: UrlFieldSpec<T[K]> }

/** Returns the raw parameter when it is one of the allowed values. */
export function pickEnum(value: string | null, allowed: readonly string[]): string | null {
  if (value === null) return null
  return allowed.includes(value) ? value : null
}

/**
 * Returns the parameter when it is a run of digits, so a field can be emptied
 * while typing without the empty string being treated as a value.
 */
export function digitsOnly(value: string | null): string | null {
  if (value === null) return null
  const trimmed = value.trim()
  return /^\d+$/.test(trimmed) ? trimmed : null
}

/** Reads a non-empty trimmed string, or null. */
export function nonEmptyText(value: string | null): string | null {
  if (value === null) return null
  const trimmed = value.trim()
  return trimmed === '' ? null : trimmed
}

/** The value each field takes when the URL supplies nothing. */
export function defaultsFromSchema<T extends object>(schema: UrlSchema<T>): T {
  const values: Record<string, unknown> = {}
  for (const key of Object.keys(schema)) {
    values[key] = schema[key as keyof T].default
  }
  return values as T
}

/**
 * Writes the values that differ from their default into a query string.
 *
 * Omitting defaults keeps the common case, a page opened with no query string,
 * from carrying a wall of redundant parameters.
 */
export function serializeState<T extends object>(
  schema: UrlSchema<T>,
  values: T,
): string {
  const params = new URLSearchParams()
  for (const key of Object.keys(schema) as Array<keyof T & string>) {
    const spec = schema[key]
    if (spec.omit) continue
    const value = values[key]
    if (Object.is(value, spec.default)) continue
    params.set(spec.param, spec.serialize ? spec.serialize(value) : String(value))
  }
  return params.toString()
}

/** Which fields the URL actually supplied, keyed by field name. */
export type SeededFlags<T> = { [K in keyof T]: boolean }

export interface UrlSyncedState<T> {
  values: T
  update: (patch: Partial<T>) => void
  /** The query string these values produce. */
  serialized: string
  /** Which fields came from the URL, so a page can avoid overriding them. */
  seeded: { readonly current: SeededFlags<T> }
}

/**
 * Binds a value object to the query string.
 *
 * The schema must be a module level constant. It is captured in a ref, so a
 * schema rebuilt on every render cannot restart the effects.
 */
export function useUrlSyncedState<T extends object>(
  schema: UrlSchema<T>,
): UrlSyncedState<T> {
  const schemaRef = useRef(schema)
  const [searchParams, setSearchParams] = useSearchParams()

  const [values, setValues] = useState<T>(() => defaultsFromSchema(schemaRef.current))
  const [ready, setReady] = useState(false)
  const seededRef = useRef<SeededFlags<T>>(
    Object.fromEntries(Object.keys(schema).map((key) => [key, false])) as SeededFlags<T>,
  )

  useEffect(() => {
    const current = schemaRef.current
    const patch: Record<string, unknown> = {}
    const nextSeeded: Record<string, boolean> = {}

    for (const key of Object.keys(current)) {
      const spec = current[key as keyof T]
      if (spec.omit) continue
      const parsed = spec.parse(searchParams.get(spec.param))
      if (parsed !== null && parsed !== undefined) {
        patch[key] = parsed
        nextSeeded[key] = true
      }
    }

    seededRef.current = { ...seededRef.current, ...nextSeeded }
    if (Object.keys(patch).length > 0) {
      setValues((existing) => ({ ...existing, ...patch }))
    }
    // Ready gates the write effect below until the URL has been applied to
    // state. Without it the write effect would run in the same commit as this
    // read, see the defaults still in state, and briefly wipe the query string.
    setReady(true)
    // The URL is read once, on mount. Later navigations are written, not read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const serialized = useMemo(() => serializeState(schemaRef.current, values), [values])

  useEffect(() => {
    if (!ready) return
    if (serialized === searchParams.toString()) return
    setSearchParams(new URLSearchParams(serialized), { replace: true, preventScrollReset: true })
  }, [ready, serialized, searchParams, setSearchParams])

  const update = useCallback((patch: Partial<T>) => {
    setValues((current) => ({ ...current, ...patch }))
  }, [])

  return { values, update, serialized, seeded: seededRef }
}
