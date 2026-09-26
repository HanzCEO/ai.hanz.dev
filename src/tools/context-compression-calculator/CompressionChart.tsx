import type { ReactNode } from 'react'

import { formatUsd } from '@/lib/format'
import type { CompressionResult } from '@/lib/compression'

/**
 * Two charts of the same decision.
 *
 * The first plots the cost of the next request against the share of the session
 * the summary keeps, so the crossing is a real crossing and the break-even
 * share is a point on the x axis. The second plots the same two costs against
 * the size of the context, where both are rays from the origin and the cheaper
 * one is simply the shallower ray.
 *
 * Both are plain inline SVG with no browser API, so a server render draws them
 * without a client pass.
 */

const VIEW_WIDTH = 720
const VIEW_HEIGHT = 340
const PAD_LEFT = 76
const PAD_RIGHT = 20
const PAD_TOP = 20
const PAD_BOTTOM = 52
const PLOT_WIDTH = VIEW_WIDTH - PAD_LEFT - PAD_RIGHT
const PLOT_HEIGHT = VIEW_HEIGHT - PAD_TOP - PAD_BOTTOM
const PLOT_BOTTOM = PAD_TOP + PLOT_HEIGHT
const PLOT_RIGHT = PAD_LEFT + PLOT_WIDTH

/** Five gridlines, from zero to the top of the axis. */
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1]

interface FrameProps {
  /** Value the top gridline stands for. */
  yMax: number
  /** Ticks along the x axis, as a fraction of the plot width and a label. */
  xTicks: Array<{ fraction: number; label: string }>
  xTitle: string
  yTitle: string
  /** Spoken description of the whole chart, which replaces the drawn shapes. */
  ariaLabel: string
  /** The lines and markers drawn inside the plot. */
  children: ReactNode
}

/** The axes, gridlines, and tick labels both charts share. */
function Frame({ yMax, xTicks, xTitle, yTitle, ariaLabel, children }: FrameProps) {
  const yFor = (value: number) => PAD_TOP + (1 - value / yMax) * PLOT_HEIGHT

  return (
    <svg
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      className="h-auto w-full"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={ariaLabel}
    >
      {Y_TICKS.map((fraction) => {
        const value = yMax * fraction
        const y = yFor(value)
        return (
          <g key={fraction}>
            <line
              x1={PAD_LEFT}
              x2={PLOT_RIGHT}
              y1={y}
              y2={y}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={PAD_LEFT - 8}
              y={y + 4}
              textAnchor="end"
              fontSize={11}
              fill="var(--muted-foreground)"
            >
              {formatUsd(value)}
            </text>
          </g>
        )
      })}

      {xTicks.map((tick) => {
        const x = PAD_LEFT + tick.fraction * PLOT_WIDTH
        return (
          <g key={tick.label}>
            <line
              x1={x}
              x2={x}
              y1={PAD_TOP}
              y2={PLOT_BOTTOM}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={x}
              y={PLOT_BOTTOM + 18}
              textAnchor="middle"
              fontSize={11}
              fill="var(--muted-foreground)"
            >
              {tick.label}
            </text>
          </g>
        )
      })}

      <line
        x1={PAD_LEFT}
        x2={PLOT_RIGHT}
        y1={PLOT_BOTTOM}
        y2={PLOT_BOTTOM}
        stroke="var(--foreground)"
        strokeWidth={1}
      />
      <line
        x1={PAD_LEFT}
        x2={PAD_LEFT}
        y1={PAD_TOP}
        y2={PLOT_BOTTOM}
        stroke="var(--foreground)"
        strokeWidth={1}
      />

      <text
        x={PAD_LEFT + PLOT_WIDTH / 2}
        y={VIEW_HEIGHT - 12}
        textAnchor="middle"
        fontSize={12}
        fill="var(--foreground)"
      >
        {xTitle}
      </text>
      <text
        x={16}
        y={PAD_TOP + PLOT_HEIGHT / 2}
        textAnchor="middle"
        fontSize={12}
        fill="var(--foreground)"
        transform={`rotate(-90 16 ${PAD_TOP + PLOT_HEIGHT / 2})`}
      >
        {yTitle}
      </text>

      {children}
    </svg>
  )
}

function Legend({ items }: { items: Array<{ label: string; tone: 'kept' | 'summary'; dashed?: boolean }> }) {
  return (
    <ul className="text-muted-foreground mt-3 flex list-none flex-wrap gap-x-5 gap-y-2 p-0 text-xs">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-2">
          <svg width={18} height={8} aria-hidden="true">
            <line
              x1={0}
              x2={18}
              y1={4}
              y2={4}
              stroke={item.tone === 'summary' ? 'var(--muted-foreground)' : 'var(--primary)'}
              strokeWidth={2}
              strokeDasharray={item.dashed ? '4 3' : undefined}
            />
          </svg>
          {item.label}
        </li>
      ))}
    </ul>
  )
}

/**
 * Cost against the share of the session the summary keeps.
 *
 * Keeping the whole session costs the same whatever the share, so it is a flat
 * line. A summary costs the full input price on every token it keeps, so its
 * line climbs from zero and crosses the flat line at the break-even share.
 */
export function CompressionShareChart({ result }: { result: CompressionResult }) {
  const summaryAtFull = result.compressRatePerMillion / (result.compressionPercent / 100)
  // The summary line climbs to the full input price at a share of 100, which on
  // a cheap cache is many times the kept line. The axis stops well short of
  // that, so the crossing stays legible, and the summary line is drawn only as
  // far as the top of the plot.
  const yMax = Math.max(
    result.neverRatePerMillion * 1.15,
    Math.min(summaryAtFull, result.neverRatePerMillion * 2.5),
  )
  const yFor = (value: number) => PAD_TOP + (1 - value / yMax) * PLOT_HEIGHT
  const xFor = (percent: number) => PAD_LEFT + (percent / 100) * PLOT_WIDTH

  const summaryEndPercent = summaryAtFull > 0 ? Math.min(100, (yMax / summaryAtFull) * 100) : 100

  const crossingX = xFor(result.breakEvenPercent)
  const crossingY = yFor(result.neverRatePerMillion)
  const crossingInRange = result.breakEvenPercent >= 0 && result.breakEvenPercent <= 100
  const labelRight = result.breakEvenPercent > 60

  return (
    <figure className="flex flex-col">
      <Frame
        yMax={yMax}
        xTitle="Share of the session the summary keeps"
        yTitle="Dollars for the next request"
        ariaLabel={`Cost of the next request against the share of the session a summary keeps. Keeping the whole session costs ${formatUsd(result.neverRatePerMillion)} and a summary keeping ${result.compressionPercent} percent costs ${formatUsd(result.compressRatePerMillion)}. The two agree at ${result.breakEvenPercent.toFixed(1)} percent.`}
        xTicks={[0, 25, 50, 75, 100].map((percent) => ({
          fraction: percent / 100,
          label: `${percent}%`,
        }))}
      >
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(result.neverRatePerMillion)}
          y2={yFor(result.neverRatePerMillion)}
          stroke="var(--primary)"
          strokeWidth={2.5}
        />
        <line
          x1={xFor(0)}
          x2={xFor(summaryEndPercent)}
          y1={yFor(0)}
          y2={yFor(Math.min(summaryAtFull, yMax))}
          stroke="var(--muted-foreground)"
          strokeWidth={2.5}
        />
        <line
          x1={xFor(result.compressionPercent)}
          x2={xFor(result.compressionPercent)}
          y1={PAD_TOP}
          y2={PLOT_BOTTOM}
          stroke="var(--muted-foreground)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        {crossingInRange && (
          <g>
            <circle cx={crossingX} cy={crossingY} r={4.5} fill="var(--primary)" />
            {/*
              The label sits below the crossing, where the space is empty. The
              rising line has already passed above it, and the flat line leaves
              the space below clear across the whole width.
            */}
            <text
              x={crossingX + (labelRight ? -8 : 8)}
              y={crossingY + 18}
              textAnchor={labelRight ? 'end' : 'start'}
              fontSize={12}
              fill="var(--primary)"
            >
              {`${result.breakEvenPercent.toFixed(1)}% ties`}
            </text>
          </g>
        )}
      </Frame>
      <Legend
        items={[
          { label: 'Keep the whole session', tone: 'kept' },
          { label: `Summarise to ${result.compressionPercent}%`, tone: 'summary' },
        ]}
      />
      <figcaption className="text-muted-foreground mt-3 text-xs">
        {`The flat line is the session you already hold, which costs the same at any share because you are not changing it. The rising line is a summary, and it costs the full input price on every token it keeps. They meet at ${result.breakEvenPercent.toFixed(1)} percent, so a summary that keeps less than that is cheaper than carrying the whole session, and a summary that keeps more is dearer. The dashed vertical line marks the share you entered.`}
      </figcaption>    </figure>
  )
}

/**
 * Cost against the size of the context, with the summary held at the share the
 * reader chose. Both paths are rays from the origin, so the cheaper one is the
 * shallower ray at every context size.
 */
export function CompressionContextChart({ result }: { result: CompressionResult }) {
  const yMax = Math.max(result.neverRatePerMillion, result.compressRatePerMillion) * 1.08
  const yFor = (value: number) => PAD_TOP + (1 - value / yMax) * PLOT_HEIGHT

  return (
    <figure className="flex flex-col">
      <Frame
        yMax={yMax}
        xTitle="Context tokens"
        yTitle="Dollars for the next request"
        ariaLabel={`Cost of the next request against the context size, up to ${result.contextWindowTokens.toLocaleString('en-US')} tokens. At the full window, keeping the session costs ${formatUsd(result.neverAtWindow)} and summarising to ${result.compressionPercent} percent costs ${formatUsd(result.compressAtWindow)}.`}
        xTicks={[0, 0.25, 0.5, 0.75, 1].map((fraction) => ({
          fraction,
          label: fraction === 0 ? '0' : `${fraction}M`,
        }))}
      >
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(0)}
          y2={yFor(result.neverRatePerMillion)}
          stroke="var(--primary)"
          strokeWidth={2.5}
        />
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(0)}
          y2={yFor(result.compressRatePerMillion)}
          stroke="var(--muted-foreground)"
          strokeWidth={2.5}
        />
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(0)}
          y2={yFor(result.neverRatePerMillion)}
          stroke="var(--primary)"
          strokeWidth={2}
          strokeDasharray="5 4"
          opacity={0.5}
        />
      </Frame>
      <Legend
        items={[
          { label: 'Keep the whole session', tone: 'kept' },
          { label: `Summarise to ${result.compressionPercent}%`, tone: 'summary' },
          {
            label: 'Tie, which is a summary at the break-even share',
            tone: 'kept',
            dashed: true,
          },
        ]}
      />
      <figcaption className="text-muted-foreground mt-3 text-xs">
        {`Both costs grow in step with the context, so the shallower line stays cheaper at every size, and the dashed line lands on the kept line because a summary at the break-even share costs exactly what keeping the session costs.`}
      </figcaption>
    </figure>
  )
}

export default function CompressionChart({ result }: { result: CompressionResult }) {
  return (
    <div className="flex flex-col gap-8">
      <CompressionShareChart result={result} />
      <CompressionContextChart result={result} />
    </div>
  )
}
