"use client";

import {
  Brush,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import type { HistoryBar } from "@/lib/types";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/types";

function ticksEveryNth<T>(arr: T[], n: number): T[] {
  return arr.filter((_, i) => i % n === 0);
}

interface RegimeBand {
  start: string;
  end: string;
  label: number;
}

function detectRegimeBands(history: HistoryBar[]): RegimeBand[] {
  if (history.length === 0) return [];
  const bands: RegimeBand[] = [];
  let curStart = history[0].date;
  let curLabel = history[0].label;
  for (let i = 1; i < history.length; i++) {
    if (history[i].label !== curLabel) {
      bands.push({ start: curStart, end: history[i].date, label: curLabel });
      curStart = history[i].date;
      curLabel = history[i].label;
    }
  }
  bands.push({
    start: curStart,
    end: history[history.length - 1].date,
    label: curLabel,
  });
  return bands;
}

export default function RegimeTimelineChart({
  history,
}: {
  history: HistoryBar[];
}) {
  const bands = useMemo(() => detectRegimeBands(history), [history]);
  const firstClose = history.find((h) => h.close !== null)?.close ?? 1;
  const data = useMemo(
    () =>
      history.map((h) => ({
        ...h,
        norm: h.close !== null ? (h.close / firstClose) * 100 : null,
      })),
    [history, firstClose],
  );
  const dates = data.map((d) => d.date);
  const tickInterval = Math.max(1, Math.floor(history.length / 8));

  // The latest bar — for the "TODAY" reference line + labeled dot
  const lastBar = data[data.length - 1];
  const lastColor = REGIME_COLORS[lastBar?.label ?? 2] ?? "#a3a3a3";

  // Default zoom: last 6 months (~126 trading days)
  const defaultStart = Math.max(0, data.length - 126);

  return (
    <div className="panel p-3">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
            REGIME TIMELINE · ZOOMABLE · LIVE THROUGH {lastBar?.date}
          </div>
          <div className="text-sm text-ink">
            Rule-baseline 3-regime overlay on normalized close. Drag the brush
            below to zoom.
          </div>
        </div>
        <Legend />
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 8, right: 60, bottom: 0, left: -8 }}
          >
            <CartesianGrid stroke="#1e2a3a" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: "#9aa6b2", fontSize: 11 }}
              ticks={ticksEveryNth(dates, tickInterval)}
              minTickGap={10}
            />
            <YAxis
              tick={{ fill: "#9aa6b2", fontSize: 11 }}
              domain={["dataMin - 2", "dataMax + 2"]}
            />
            {bands.map((b, i) => (
              <ReferenceArea
                key={i}
                x1={b.start}
                x2={b.end}
                ifOverflow="hidden"
                fill={REGIME_COLORS[b.label]}
                fillOpacity={0.12}
                stroke="none"
              />
            ))}
            <Line
              type="monotone"
              dataKey="norm"
              name="Close (norm=100)"
              stroke="#e6edf3"
              strokeWidth={1.6}
              dot={false}
              activeDot={{ r: 4, fill: "#e6edf3", stroke: "#2E75B6", strokeWidth: 2 }}
              isAnimationActive={false}
            />
            {lastBar && (
              <ReferenceLine
                x={lastBar.date}
                stroke={lastColor}
                strokeDasharray="2 4"
                strokeWidth={1.5}
                label={{
                  value: `TODAY · ${lastBar.regime}`,
                  position: "top",
                  fill: lastColor,
                  fontSize: 10,
                  fontFamily: "JetBrains Mono, monospace",
                  offset: 4,
                }}
                ifOverflow="visible"
              />
            )}
            <Tooltip content={<RegimeTip />} />
            <Brush
              dataKey="date"
              height={22}
              stroke="#2E75B6"
              fill="#0f1521"
              travellerWidth={8}
              startIndex={defaultStart}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {[0, 1, 2].map((r) => (
        <span
          key={r}
          className="inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider"
          style={{
            backgroundColor: `${REGIME_COLORS[r]}1a`,
            color: REGIME_COLORS[r],
          }}
        >
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: REGIME_COLORS[r] }}
          />
          {REGIME_NAMES[r]}
        </span>
      ))}
    </div>
  );
}

interface TipPayload {
  active?: boolean;
  label?: string;
  payload?: Array<{ payload: HistoryBar & { norm: number | null } }>;
}

function RegimeTip(props: TipPayload) {
  if (!props.active || !props.payload || props.payload.length === 0) return null;
  const p = props.payload[0].payload;
  const color = REGIME_COLORS[p.label] ?? "#a3a3a3";
  return (
    <div className="rounded border border-bg-ring bg-bg-panel/95 p-2 font-mono text-xs shadow-2xl backdrop-blur">
      <div className="mb-1 text-ink-muted">{p.date}</div>
      <div className="text-ink">
        CLOSE <span className="text-accent-lblue">{p.close?.toFixed(2) ?? "—"}</span>
      </div>
      <div style={{ color }}>
        {p.regime} {p.alloc !== null ? `(${(p.alloc * 100).toFixed(0)}%)` : ""}
      </div>
      <div className="text-ink-muted">
        TVTP {p.tvtp_pos !== null ? (p.tvtp_pos >= 0 ? "+" : "") + (p.tvtp_pos * 100).toFixed(0) + "%" : "—"}
      </div>
    </div>
  );
}