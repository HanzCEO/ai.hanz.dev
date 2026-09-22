import { useCallback, useState } from 'react'

/**
 * Which step of the layered calculator is open.
 *
 * The page asks for the cache first and the hardware second, so the first step
 * is open on arrival. The second step opens on its own once the cache has a
 * size, which is the hand-off a reader expects.
 *
 * Step one stays open by default, so a seeded run still shows the model, the
 * context and the sequences the reader arrived with. A reader's own choice is
 * kept as an override, so folding the step away is never undone by the next
 * render. An effect would fight the reader, and would also be a second render.
 */
export function useLayers(hasCacheResult: boolean) {
  const [cacheChoice, setCacheChoice] = useState<boolean | null>(null)
  const [gpuChoice, setGpuChoice] = useState<boolean | null>(null)

  const setCacheOpen = useCallback((open: boolean) => setCacheChoice(open), [])
  const setGpuOpen = useCallback((open: boolean) => setGpuChoice(open), [])

  return {
    cacheOpen: cacheChoice ?? true,
    gpuOpen: gpuChoice ?? hasCacheResult,
    setCacheOpen,
    setGpuOpen,
  }
}
