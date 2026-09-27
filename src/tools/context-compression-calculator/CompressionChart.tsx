import type { ReactNode } from 'react'

import { formatTokensShort, formatUsd } from '@/lib/format'
import type { CompressionResult } from '@/lib/compression'

/**
 * Two views of the same decision.
 *
 * The first is the answer. It plots the cost of the next request against the
 * size of the session, where keeping the session climbs from the origin and a
 * summary sits flat, so the two cross at one session size. That crossing point
 * is when to compress.
 *
 * The second explains the cap. It plots both costs against the share of the
 * window the summary is capped at, with the session held at a full window, so
 * the reader can see how far the cap can go before it stops paying.
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

/**
 * Rounds an axis top up so its quarters land on round numbers.
 *
 * The axis is drawn as five gridlines, so the top value decides every label on
 * it. Raw data gives tops like 0.5725, which labels the gridlines as 0.143 and
 * 0.286, and a reader cannot check those against anything. Rounding the top to
 * something whose quarters divide evenly keeps every label readable.
 */
function niceAxisMax(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1

  const rough = value / 4
  const magnitude = 10 ** Math.floor(Math.log10(rough))

  for (const candidate of [1, 2, 2.5, 5, 10]) {
    if (candidate * magnitude >= rough) return candidate * magnitude * 4
  }
  return 10 * magnitude * 4
}

const KEPT_STROKE = 'var(--primary)'
const SUMMARY_STROKE = 'var(--muted-foreground)'

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
              stroke={item.tone === 'summary' ? SUMMARY_STROKE : KEPT_STROKE}
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
 * The answer, as a picture: cost against the size of the session.
 *
 * Keeping the session costs the same for every token of context, so its line
 * climbs from the origin. A summary is capped at a share of the window, so it
 * costs the same whatever the session holds, and its line is flat. The flat
 * line starts below the climbing line and is overtaken at one session size,
 * which is the size where compressing starts to pay.
 */
export function CompressionSizeChart({ result }: { result: CompressionResult }) {
  const yMax = niceAxisMax(Math.max(result.keptAtWindow, result.summaryCost))
  const yFor = (value: number) => PAD_TOP + (1 - value / yMax) * PLOT_HEIGHT
  const xFor = (fraction: number) => PAD_LEFT + fraction * PLOT_WIDTH

  const crossingFraction = result.breakEvenSessionTokens / result.contextWindowTokens
  const crossingInRange = crossingFraction > 0 && crossingFraction <= 1
  const crossingX = xFor(Math.min(crossingFraction, 1))
  const crossingY = yFor(result.summaryCost)
  // Below the flat line and right of the crossing is empty, because the climbing
  // line has already passed above it. Near the right edge there is no room for a
  // label there, so it goes above the flat line and left of the crossing, which
  // is empty for the same reason in reverse.
  const labelRight = crossingFraction <= 0.75

  return (
    <figure className="flex flex-col">
      <Frame
        yMax={yMax}
        xTitle="Session size, in tokens of context"
        yTitle="Dollars for the next request"
        ariaLabel={
          crossingInRange
            ? `Cost of the next request against the size of the session. Keeping the session costs ${formatUsd(result.keptAtWindow)} at a full window, and carrying a ${result.compressionPercent} percent summary costs ${formatUsd(result.summaryCost)} at any size. The two agree at ${formatTokensShort(result.breakEvenSessionTokens)} tokens, so compressing pays past that size.`
            : `Cost of the next request against the size of the session. Keeping the session costs ${formatUsd(result.keptAtWindow)} at a full window, and carrying a ${result.compressionPercent} percent summary costs ${formatUsd(result.summaryCost)} at any size. The summary never becomes the cheaper of the two inside the window.`
        }
        xTicks={[0, 0.25, 0.5, 0.75, 1].map((fraction) => ({
          fraction,
          label: fraction === 0 ? '0' : `${formatTokensShort(fraction * result.contextWindowTokens)}`,
        }))}
      >
        {/* Keeping the session, which climbs from the origin to the full window. */}
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(0)}
          y2={yFor(result.keptAtWindow)}
          stroke={KEPT_STROKE}
          strokeWidth={2.5}
        />
        {/* Carrying the summary, which is the same at every session size. */}
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(result.summaryCost)}
          y2={yFor(result.summaryCost)}
          stroke={SUMMARY_STROKE}
          strokeWidth={2.5}
        />
        {crossingInRange && (
          <g>
            <line
              x1={crossingX}
              x2={crossingX}
              y1={crossingY}
              y2={PLOT_BOTTOM}
              stroke={SUMMARY_STROKE}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <circle cx={crossingX} cy={crossingY} r={4.5} fill={KEPT_STROKE} />
            {/* The label sits clear of both lines, in whichever quarter is empty. */}
            <text
              x={crossingX + (labelRight ? 10 : -10)}
              y={labelRight ? crossingY + 18 : crossingY - 12}
              textAnchor={labelRight ? 'start' : 'end'}
              fontSize={12}
              fill={KEPT_STROKE}
            >
              {`${formatTokensShort(result.breakEvenSessionTokens)} tokens ties`}
            </text>
          </g>
        )}
      </Frame>
      <Legend
        items={[
          { label: 'Keep the whole session', tone: 'kept' },
          {
            label: `Carry a ${result.compressionPercent}% summary`,
            tone: 'summary',
          },
        ]}
      />
      <figcaption className="text-muted-foreground mt-3 text-xs">
        {`The climbing line is the session you already hold, which gets dearer with every token you add. The flat line is a summary capped at ${result.compressionPercent} percent of the window, which costs the same however long the session runs. `}
        {crossingInRange
          ? 'Where they cross is the size at which compressing starts to pay.'
          : 'The flat line stays above the climbing line for the whole window, so no session size makes compressing pay.'}
      </figcaption>
    </figure>
  )
}

/**
 * The cap, as a picture: cost against the share of the window the summary keeps.
 *
 * The session is held at a full window here, so keeping it is one flat price.
 * A wider cap means more summary tokens at the full input price, so the summary
 * line climbs and crosses the flat line at the share where it stops paying.
 */
export function CompressionShareChart({ result }: { result: CompressionResult }) {
  // A cheap cache makes the summary line far steeper than the kept line, so the
  // axis only has to reach far enough to show the crossing clearly.
  const yMax = niceAxisMax(Math.max(result.keptAtWindow * 1.3, result.summaryCost * 2.2))
  const yFor = (value: number) => PAD_TOP + (1 - value / yMax) * PLOT_HEIGHT
  const xFor = (percent: number) => PAD_LEFT + (percent / 100) * PLOT_WIDTH

  // Dollars per percent of the window, so the line can be drawn to the cap the
  // reader set rather than to the top of the axis.
  const costPerPercent = result.summaryUnitPerMillion / 100
  const summaryEndPercent =
    costPerPercent > 0 ? Math.min(100, yMax / costPerPercent) : 100
  const crossingInRange = result.breakEvenPercent > 0 && result.breakEvenPercent <= 100
  const crossingX = xFor(result.breakEvenPercent)
  const crossingY = yFor(result.keptAtWindow)
  const labelRight = result.breakEvenPercent > 60

  return (
    <figure className="flex flex-col">
      <Frame
        yMax={yMax}
        xTitle="Summary cap, as a share of the window"
        yTitle="Dollars for the next request"
        ariaLabel={
          crossingInRange
            ? `Cost of the next request against the share of the window a summary is capped at, with the session held at a full window. Keeping the session costs ${formatUsd(result.keptAtWindow)}. The summary costs the same as keeping it at a cap of ${result.breakEvenPercent.toFixed(1)} percent, so a tighter cap pays.`
            : `Cost of the next request against the share of the window a summary is capped at, with the session held at a full window. Keeping the session costs ${formatUsd(result.keptAtWindow)}, which is more than a summary costs at any cap, so every cap pays off.`
        }
        xTicks={[0, 25, 50, 75, 100].map((percent) => ({
          fraction: percent / 100,
          label: `${percent}%`,
        }))}
      >
        <line
          x1={PAD_LEFT}
          x2={PLOT_RIGHT}
          y1={yFor(result.keptAtWindow)}
          y2={yFor(result.keptAtWindow)}
          stroke={KEPT_STROKE}
          strokeWidth={2.5}
        />
        <line
          x1={xFor(0)}
          x2={xFor(summaryEndPercent)}
          y1={yFor(0)}
          y2={yFor(Math.min(summaryEndPercent * costPerPercent, yMax))}
          stroke={SUMMARY_STROKE}
          strokeWidth={2.5}
        />
        {/* The cap the reader set, so the two charts can be read together. */}
        <line
          x1={xFor(result.compressionPercent)}
          x2={xFor(result.compressionPercent)}
          y1={PAD_TOP}
          y2={PLOT_BOTTOM}
          stroke={SUMMARY_STROKE}
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <text
          x={xFor(result.compressionPercent)}
          y={PAD_TOP + 12}
          textAnchor={result.compressionPercent > 85 ? 'end' : 'middle'}
          fontSize={11}
          fill={SUMMARY_STROKE}
        >
          {`your cap, ${result.compressionPercent}%`}
        </text>
        {crossingInRange && (
          <g>
            <circle cx={crossingX} cy={crossingY} r={4.5} fill={KEPT_STROKE} />
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
              fill={KEPT_STROKE}
            >
              {`${result.breakEvenPercent.toFixed(1)}% ties`}
            </text>
          </g>
        )}
      </Frame>
      <Legend
        items={[
          { label: 'Keep the whole session', tone: 'kept' },
          { label: 'A summary, at the cap on the axis', tone: 'summary' },
          ...(crossingInRange
            ? [
                {
                  label: 'Tie, which is a summary at the break-even cap',
                  tone: 'kept' as const,
                  dashed: true,
                },
              ]
            : []),
        ]}
      />
      <figcaption className="text-muted-foreground mt-3 text-xs">
        {`The flat line is the session you already hold, priced at a full window. The rising line is a summary, and it costs the full input price on every token it keeps, so a wider cap costs more. `}
        {crossingInRange
          ? `They meet at ${result.breakEvenPercent.toFixed(1)} percent, which is the widest cap that still pays off, and the dashed line marks the cap you entered.`
          : 'The rising line stays below the flat line across a full cap, so every cap up to 100 percent pays off.'}
        {crossingInRange &&
          summaryEndPercent < 100 &&
          ` Past a cap of ${Math.floor(summaryEndPercent)} percent the summary line leaves the top of the axis, since the answer there is already settled.`}
      </figcaption>
    </figure>
  )
}

export default function CompressionChart({ result }: { result: CompressionResult }) {
  return (
    <div className="flex flex-col gap-8">
      <CompressionSizeChart result={result} />
      <CompressionShareChart result={result} />
    </div>
  )
}
