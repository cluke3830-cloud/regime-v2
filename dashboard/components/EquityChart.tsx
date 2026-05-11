"use client";

import {
  Brush,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import type { HistoryBar } from "@/lib/types";

function ticksEveryNth<T>(arr: T[], n: number): T[] {
  return arr.filter((_, i) => i % n === 0);
}

export default function EquityChart({
  history,
}: {
  history: HistoryBar[];
}) {
  const data = useMemo(
    () =>
      history.map((h) => ({
        date: h.date,
        tvtp: h.eq_tvtp,
        bull3x: h.eq_bull3x,
        bh: h.eq_bh,
        bull3x_in: h.bull3x_in,
      })),
    [history],
  );
  const dates = data.map((d) => d.date);
  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  // Final equity values for the legend annotation
  const last = data[data.length - 1];
  const fmtMult = (v: number | null | undefined) =>
    v === null || v === undefined ? "—" : `${v.toFixed(2)}×`;

  return (
    <div className="panel p-3">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
            EQUITY CURVES — 2Y WALK-FORWARD
          </div>
          <div className="text-sm text-ink">
            TVTP-MSAR · 3× Bull Bet (rule p₀+p₁ &gt; 50%) · Buy &amp; Hold
          </div>
        </div>
        <div className="flex gap-3 font-mono text-[11px]">
          <span style={{ color: "#2E75B6" }}>
            TVTP {fmtMult(last?.tvtp)}
          </span>
          <span style={{ color: "#f5a623" }}>
            3×BULL {fmtMult(last?.bull3x)}
          </span>
          <span className="text-ink-muted">B&amp;H {fmtMult(last?.bh)}</span>
        </div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 6, right: 12, bottom: 0, left: -8 }}
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
              tickFormatter={(v: number) => `${v.toFixed(1)}×`}
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
              formatter={(value: number, name: string, item) => {
                const tag = item?.payload?.bull3x_in === 1 ? " · IN" : " · CASH";
                if (name === "3× Bull Bet") {
                  return [`${value?.toFixed(3)}×${tag}`, name];
                }
                return [`${value?.toFixed(3)}×`, name];
              }}
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
              dataKey="bull3x"
              name="3× Bull Bet"
              stroke="#f5a623"
              strokeWidth={1.6}
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
            <Brush
              dataKey="date"
              height={20}
              stroke="#2E75B6"
              fill="#0f1521"
              travellerWidth={8}
              startIndex={Math.max(0, data.length - 252)}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 font-mono text-[10px] leading-relaxed text-ink-dim">
        <span className="text-accent-amber">3× Bull Bet</span>: when the rule
        classifier puts &gt;50% probability mass on (Full Bull + Half Bull),
        go long a hypothetical 3× leveraged equity ETF (UPRO for SPY, TQQQ
        for QQQ, etc.). Else cash. Daily compounding of 3 × simple-return —
        captures the vol drag real LETFs experience. Signal lagged by 1 bar
        (no look-ahead).
      </p>
    </div>
  );
}