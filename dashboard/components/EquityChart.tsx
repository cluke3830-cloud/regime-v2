"use client";

import {
  Line,
  LineChart,
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

export default function EquityChart({
  history,
}: {
  history: HistoryBar[];
}) {
  const data = history.map((h) => ({
    date: h.date,
    tvtp: h.eq_tvtp,
    rule: h.eq_rule,
    bh: h.eq_bh,
  }));
  const dates = data.map((d) => d.date);
  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  return (
    <div className="panel p-3">
      <div className="mb-2">
        <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
          EQUITY CURVES — 2Y WALK-FORWARD
        </div>
        <div className="text-sm text-ink">
          TVTP-MSAR (champion) vs rule-baseline vs buy-and-hold
        </div>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: -8 }}>
            <CartesianGrid stroke="#1e2a3a" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: "#9aa6b2", fontSize: 11 }}
              ticks={ticksEveryNth(dates, tickInterval)}
              minTickGap={10}
            />
            <YAxis
              tick={{ fill: "#9aa6b2", fontSize: 11 }}
              tickFormatter={(v: number) => v.toFixed(2)}
              domain={["dataMin - 0.05", "dataMax + 0.05"]}
            />
            <Tooltip
              contentStyle={{
                background: "#0f1521",
                border: "1px solid #1e2a3a",
                borderRadius: 4,
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 11,
              }}
              labelStyle={{ color: "#9aa6b2" }}
              formatter={(value: number) => [value?.toFixed(3), null]}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
              iconType="line"
            />
            <Line
              type="monotone"
              dataKey="tvtp"
              name="TVTP-MSAR"
              stroke="#2E75B6"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="rule"
              name="Rule baseline"
              stroke="#f5a623"
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="bh"
              name="Buy & Hold"
              stroke="#9aa6b2"
              strokeWidth={1.2}
              strokeDasharray="3 3"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}