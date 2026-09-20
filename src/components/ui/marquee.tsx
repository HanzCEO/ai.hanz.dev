import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { cn } from 'cn'

interface MarqueeProps {
  children: ReactNode
  className?: string
  /** Pixels per second the line travels while it is being read. */
  speed?: number
}

/** Seconds, so a short overflow still moves and a long one does not crawl. */
const MIN_DURATION = 2
const MAX_DURATION = 30
/** A pixel of rounding is not overflow worth animating. */
const SLACK = 1

/**
 * Reveals a line that is wider than the box holding it by sliding it sideways.
 *
 * Dropdown rows are capped so a long recipe note cannot stretch the popup, but a
 * capped row still has to be readable. The track travels while the row is
 * hovered or highlighted and holds still otherwise, so an open list does not
 * turn into a wall of moving text.
 */
export default function Marquee({ children, className, speed = 60 }: MarqueeProps) {
  const viewportRef = useRef<HTMLSpanElement>(null)
  const trackRef = useRef<HTMLSpanElement>(null)
  const [overflow, setOverflow] = useState(0)

  useEffect(() => {
    const viewport = viewportRef.current
    const track = trackRef.current
    if (!viewport || !track) return

    const measure = () => setOverflow(track.scrollWidth - viewport.clientWidth)

    measure()

    // The popup is laid out after the first paint, and a resize changes how much
    // of the line fits, so both boxes are watched rather than measured once.
    const observer = new ResizeObserver(measure)
    observer.observe(viewport)
    observer.observe(track)
    return () => observer.disconnect()
  }, [])

  const overflows = overflow > SLACK
  const duration = Math.min(MAX_DURATION, Math.max(MIN_DURATION, overflow / speed))

  return (
    <span
      ref={viewportRef}
      data-slot="marquee"
      data-overflow={overflows ? 'true' : undefined}
      className={cn('block min-w-0 overflow-hidden', className)}
      style={
        {
          '--marquee-distance': `${Math.max(0, overflow)}px`,
          '--marquee-duration': `${duration}s`,
        } as CSSProperties
      }
    >
      <span ref={trackRef} data-slot="marquee-inner" className="block w-max whitespace-nowrap">
        {children}
      </span>
    </span>
  )
}
