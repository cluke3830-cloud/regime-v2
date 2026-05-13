import Link from "next/link";
import type { AssetSummaryItem } from "@/lib/types";
import {
  REGIME_ALLOC,
  REGIME_COLORS,
  REGIME_NAMES,
  activeRegimeFromProbs,
} from "@/lib/types";
import RegimeBadge from "./RegimeBadge";

function fmtSharpe(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  return (s >= 0 ? "+" : "") + s.toFixed(2);
}

function fmtPct(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  return (s * 100).toFixed(1) + "%";
}

export default function AssetGrid({ assets }: { assets: AssetSummaryItem[] }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {assets.map((a) => {
        // Derive the displayed regime from argmax(probs) so the card matches
        // the per-asset probability stack (rule-baseline hard label can lag).
        const activeLabel = activeRegimeFromProbs(
          a.regime.probs,
          a.regime.label,
        );
        const activeName = REGIME_NAMES[activeLabel];
        const activeAlloc = REGIME_ALLOC[activeLabel];
        const accent = REGIME_COLORS[activeLabel] ?? "#a3a3a3";
        return (
          <Link
            key={a.safe}
            href={`/asset/${a.safe}`}
            className="panel panel-glow group relative overflow-hidden p-4 transition-transform hover:-translate-y-0.5"
            style={{
              boxShadow: `inset 0 0 0 1px ${accent}22`,
            }}
          >
            <div
              className="pointer-events-none absolute -top-12 -right-12 h-24 w-24 rounded-full opacity-20 blur-2xl"
              style={{ backgroundColor: accent }}
            />
            <div className="flex items-start justify-between">
              <div>
                <div className="font-mono text-lg font-bold text-ink">
                  {a.ticker}
                </div>
                <div className="text-xs text-ink-muted">{a.name}</div>
              </div>
              <RegimeBadge
                label={activeLabel}
                name={activeName}
                alloc={activeAlloc}
                size="sm"
              />
            </div>

            <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <div>
                <div className="text-ink-dim">CLOSE</div>
                <div className="font-mono font-medium text-ink">
                  {a.close !== null ? a.close.toFixed(2) : "—"}
                </div>
              </div>
              <div>
                <div className="text-ink-dim">TVTP POS</div>
                <div
                  className="font-mono font-medium"
                  style={{
                    color: (a.tvtp?.position ?? 0) >= 0 ? "#22c55e" : "#ef4444",
                  }}
                >
                  {a.tvtp?.position !== undefined
                    ? (a.tvtp.position >= 0 ? "+" : "") +
                      (a.tvtp.position * 100).toFixed(0) +
                      "%"
                    : "—"}
                </div>
              </div>
              <div>
                <div className="text-ink-dim">SHARPE p50</div>
                <div className="font-mono font-medium text-accent-lblue">
                  {fmtSharpe(a.sharpe_p50)}
                </div>
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between border-t border-bg-ring pt-2">
              <span className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
                AS OF {a.as_of}
              </span>
              <span className="font-mono text-[10px] text-ink-dim">
                DD {fmtPct(a.max_dd_p50)}
              </span>
            </div>
          </Link>
        );
      })}
    </div>
  );
}
