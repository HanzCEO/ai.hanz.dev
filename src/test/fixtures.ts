import { readFileSync } from 'node:fs'
import path from 'node:path'

import type { RawConfig } from '@/lib/model-config'

/**
 * Model config fixtures, shared by every engine test.
 *
 * The configs are real published ones, kept in one place so a new calculator
 * does not have to reach into another tool's folder for a sample model, and so
 * the same repo id means the same file in every suite.
 */

const CONFIG_DIR = path.join(__dirname, 'fixtures', 'configs')

/** Absolute path to a config fixture, by repo name without the .json suffix. */
export function configFixturePath(name: string): string {
  return path.join(CONFIG_DIR, `${name}.json`)
}

/** Reads a config fixture and parses it, ready to hand to an engine. */
export function loadConfigFixture(name: string): RawConfig {
  return JSON.parse(readFileSync(configFixturePath(name), 'utf8')) as RawConfig
}

/** Reads a config fixture as raw text, for tests that exercise the paste path. */
export function readConfigFixtureText(name: string): string {
  return readFileSync(configFixturePath(name), 'utf8')
}
