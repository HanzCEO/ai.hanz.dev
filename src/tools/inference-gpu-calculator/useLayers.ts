import { useCallback, useState } from 'react'

/**
 * Which step of the layered calculator is open.
 *
 * The page asks for the cache first and the hardware second, so the first step
 * opens on arrival. Once the cache has a size the first step folds away and the
 * second opens on its own, which is the hand-off a reader expects.
 *
 * The default is derived from whether the cache has a size, and a reader's own
 * choice is kept as an override. That way the hand-off happens exactly once, and
 * reopening the first step to change the context is never undone by the next
 * render. An effect would fight the reader, and would also be a second render.
 */
export function useLayers(hasCacheResult: boolean) {
  const [cacheChoice, setCacheChoice] = useState<boolean | null>(null)
  const [gpuChoice, setGpuChoice] = useState<boolean | null>(null)

  const setCacheOpen = useCallback((open: boolean) => setCacheChoice(open), [])
  const setGpuOpen = useCallback((open: boolean) => setGpuChoice(open), [])

  return {
    cacheOpen: cacheChoice ?? !hasCacheResult,
    gpuOpen: gpuChoice ?? hasCacheResult,
    setCacheOpen,
    setGpuOpen,
  }
}
