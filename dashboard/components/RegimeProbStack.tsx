"use client";

import {
  Area,
  AreaChart,
  Brush,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistoryBar } from "@/lib/types";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/types";

function ticksEveryNth<T>(arr: T[], n: number): T[] {
  return arr.filter((_, i) => i % n === 0);
}

export default function RegimeProbStack({
  history,
}: {
  history: HistoryBar[];
}) {
  const data = history.map((h) => ({
    date: h.date,
    p0: h.p0,
    p1: h.p1,
    p2: h.p2,
  }));
  const dates = data.map((d) => d.date);
  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  const lastDate = history[history.length - 1]?.date ?? "—";
  const defaultStart = Math.max(0, data.length - 126);

  return (
    <div className="panel p-3">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
            REGIME PROBABILITY STACK
          </div>
          <div className="text-sm text-ink">
            Posterior P(Bull / Neutral / Bear | features) — stacked to 1.0
          </div>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider">
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent-green" />
          <span className="text-accent-green">LIVE · {lastDate}</span>
        </div>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: -8 }}>
            <CartesianGrid stroke="#1e2a3a" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: "#9aa6b2", fontSize: 11 }}
              ticks={ticksEveryNth(dates, tickInterval)}
              minTickGap={10}
            />
            <YAxis
              tick={{ fill: "#9aa6b2", fontSize: 11 }}
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                background: "#0f1521",
                border: "1px solid #1e2a3a",
                borderRadius: 4,
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 11,
              }}
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
            {[0, 1, 2].map((r) => (
              <Area
                key={r}
                type="monotone"
                dataKey={`p${r}`}
                name={REGIME_NAMES[r]}
                stackId="1"
                stroke={REGIME_COLORS[r]}
                fill={REGIME_COLORS[r]}
                fillOpacity={0.55}
                isAnimationActive={false}
              />
            ))}
            <Brush
              dataKey="date"
              height={20}
              stroke="#2E75B6"
              fill="#0f1521"
              travellerWidth={8}
              startIndex={defaultStart}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
