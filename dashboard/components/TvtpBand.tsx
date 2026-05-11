"use client";

import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from "recharts";
import type { HistoryBar } from "@/lib/types";

function ticksEveryNth<T>(arr: T[], n: number): T[] {
  return arr.filter((_, i) => i % n === 0);
}

export default function TvtpBand({
  history,
}: {
  history: HistoryBar[];
}) {
  const data = history.map((h) => ({
    date: h.date,
    low: h.tvtp_low,
    high: h.tvtp_high,
  }));
  const dates = data.map((d) => d.date);
  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  return (
    <div className="panel p-3">
      <div className="mb-2">
        <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
          TVTP-MSAR HAMILTON FILTER
        </div>
        <div className="text-sm text-ink">
          P(Low-Vol Bull) vs P(High-Vol Stress) — 2-state Markov-Switching AR(1)
        </div>
      </div>
      <div className="h-48">
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
            <Area
              type="monotone"
              dataKey="low"
              name="Low-Vol Bull (+100%)"
              stackId="t"
              stroke="#22c55e"
              fill="#22c55e"
              fillOpacity={0.55}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="high"
              name="High-Vol Stress (-30%)"
              stackId="t"
              stroke="#ef4444"
              fill="#ef4444"
              fillOpacity={0.55}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
