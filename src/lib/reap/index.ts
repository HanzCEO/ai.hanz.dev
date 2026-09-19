export { attentionParamsPerLayer } from './attention'
export { estimateReap } from './compute'
export { REAP_FAQ } from './faq'
export { detectMoeShape } from './moe-shape'
export {
  CALIBRATION_PRESETS,
  DEFAULT_CALIBRATION_PRESET,
  DEFAULT_MFU,
  DEFAULT_MICRO_BATCH,
  DEFAULT_OVERHEAD_FACTOR,
  DEFAULT_PRUNE_RATIO,
  DEFAULT_SETUP_SECONDS,
  PRUNE_RATIOS,
  RUNTIME_OVERHEAD_BYTES,
  WEIGHT_DTYPES,
  WEIGHT_DTYPE_ORDER,
  bytesPerParam,
  findCalibrationPreset,
  type CalibrationPreset,
  type WeightDtypeSpec,
} from './presets'
export {
  ReapInputError,
  type MoeShape,
  type ReapBound,
  type ReapInputs,
  type ReapResult,
  type ReapVerdict,
  type WeightDtype,
} from './types'
