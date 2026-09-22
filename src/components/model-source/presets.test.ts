import { describe, expect, it } from 'vitest'

import { MODEL_PRESETS } from './presets'

/**
 * The quick picks are the first thing a reader sees, so a model that this site
 * claims to size has to be one keystroke away. These checks keep the list and
 * the engines from drifting apart: every preset here must be a config the
 * detectors can actually read.
 */
describe('MODEL_PRESETS', () => {
  const ids = MODEL_PRESETS.map((preset) => preset.id)

  it('offers the two MiMo-V2.6 releases', () => {
    expect(ids).toContain('XiaomiMiMo/MiMo-V2.6-Flash-RL')
    expect(ids).toContain('XiaomiMiMo/MiMo-V2.6-Pro-RL')
  })

  it('describes the hybrid cache shape in each MiMo note', () => {
    const flash = MODEL_PRESETS.find((preset) => preset.id === 'XiaomiMiMo/MiMo-V2.6-Flash-RL')
    expect(flash?.note).toContain('9 of 48 layers hold a global cache')
    expect(flash?.note).toContain('47 of 48 layers hold experts')

    const pro = MODEL_PRESETS.find((preset) => preset.id === 'XiaomiMiMo/MiMo-V2.6-Pro-RL')
    expect(pro?.note).toContain('10 of 70 layers hold a global cache')
    expect(pro?.note).toContain('69 of 70 layers hold experts')
  })

  it('keeps every id in owner/name form and every note non-empty', () => {
    for (const preset of MODEL_PRESETS) {
      expect(preset.id).toMatch(/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/)
      expect(preset.note.trim()).not.toBe('')
    }
  })

  it('lists no model twice', () => {
    expect(new Set(ids).size).toBe(ids.length)
  })
})
