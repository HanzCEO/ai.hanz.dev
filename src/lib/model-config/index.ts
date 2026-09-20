export {
  dtypeNameToBytes,
  readArrayFrom,
  readBoolean,
  readNumber,
  readNumberArray,
  readNumberFrom,
  readObject,
  readString,
  readStringArray,
  readStringFrom,
  unwrapConfig,
} from './readers'
export { ConfigParseError, parseConfigText } from './parse'
export { MANUAL_DEFAULTS, manualConfig, type ManualShapeInputs } from './manual'
export type { RawConfig } from './types'
