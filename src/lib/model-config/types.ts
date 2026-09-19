/**
 * A model config as returned by a hub or pasted by a user. Fields are read
 * defensively everywhere, so the shape is a plain record rather than a typed
 * interface: no field is guaranteed to exist.
 */
export type RawConfig = Record<string, unknown>
