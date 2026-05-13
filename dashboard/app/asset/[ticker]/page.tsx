import { notFound } from "next/navigation";
import Link from "next/link";
import { loadAsset, loadAssetIndex, loadSummary } from "@/lib/data";
import TopBar from "@/components/TopBar";
import RegimeBadge from "@/components/RegimeBadge";
import RegimeTimelineChart from "@/components/RegimeTimelineChart";
import EquityChart from "@/components/EquityChart";
import RegimeProbStack from "@/components/RegimeProbStack";
import TvtpBand from "@/components/TvtpBand";
import TransitionHeatmap from "@/components/TransitionHeatmap";
import {
  REGIME_ALLOC,
  REGIME_COLORS,
  REGIME_NAMES,
  confirmedActiveLabelFromAsset,
} from "@/lib/types";

export const dynamicParams = false;

export async function generateStaticParams() {
  const index = await loadAssetIndex();
  return index.universe.map((u) => ({ ticker: u.safe }));
}

function fmt(s: number | null | undefined, dp = 2): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  return s.toFixed(dp);
}

function fmtSigned(s: number | null | undefined, dp = 2): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  return (s >= 0 ? "+" : "") + s.toFixed(dp);
}

function fmtPct(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  return (s * 100).toFixed(1) + "%";
}

export default async function AssetPage({
  params,
}: {
  params: { ticker: string };
}) {
  let asset, index, summary;
  try {
    [asset, index, summary] = await Promise.all([
      loadAsset(params.ticker),
      loadAssetIndex(),
      loadSummary(),
    ]);
  } catch {
    notFound();
  }

  const tvtpPos = asset.current_tvtp.position;
  const tvtpColor = tvtpPos >= 0 ? "#22c55e" : "#ef4444";

  // 2-of-3 confirmation across rule-baseline argmax, GMM label, and TVTP
  // 2-state. Flips ~1 day earlier than rule-baseline alone when two of the
  // three models agree first, while needing two votes to switch keeps
  // whipsaw modest.
  const activeLabel = confirmedActiveLabelFromAsset(asset);
  const activeName = REGIME_NAMES[activeLabel];
  const activeAlloc = REGIME_ALLOC[activeLabel];

  // pct change over last 21 bars
  const tail = asset.history.slice(-22);
  const startClose = tail[0]?.close ?? null;
  const endClose = tail[tail.length - 1]?.close ?? null;
  const monthRet =
    startClose && endClose ? endClose / startClose - 1 : null;

  return (
    <main className="min-h-screen">
      <TopBar
        universe={index.universe}
        generatedAt={summary.generated_at}
      />

      <section className="mx-auto max-w-7xl px-4 py-6">
        <div className="mb-1 flex items-center gap-2">
          <Link
            href="/"
            className="font-mono text-[11px] uppercase tracking-wider text-ink-dim hover:text-accent-lblue"
          >
            ← UNIVERSE
          </Link>
          <span className="text-ink-dim">/</span>
          <span className="font-mono text-[11px] uppercase tracking-wider text-ink-muted">
            {asset.ticker}
          </span>
        </div>

        <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="flex items-baseline gap-3">
              <h1 className="font-mono text-4xl font-bold tracking-tight text-ink">
                {asset.ticker}
              </h1>
              <span className="text-base text-ink-muted">{asset.name}</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <RegimeBadge
                label={activeLabel}
                name={activeName}
                alloc={activeAlloc}
                size="lg"
              />
              <span className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
                AS OF {asset.as_of}
              </span>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Stat
              label="LAST CLOSE"
              value={fmt(asset.current_close)}
              accent="#e6edf3"
            />
            <Stat
              label="21D RETURN"
              value={fmtSigned(monthRet ? monthRet * 100 : null) + "%"}
              accent={(monthRet ?? 0) >= 0 ? "#22c55e" : "#ef4444"}
            />
            <Stat
              label="TVTP POSITION"
              value={fmtSigned(tvtpPos * 100, 0) + "%"}
              accent={tvtpColor}
              hint={asset.current_tvtp.state}
            />
            <Stat
              label="CPCV SHARPE p50"
              value={fmtSigned(asset.stats.sharpe_p50)}
              accent="#2E75B6"
              hint="45 OOS PATHS"
            />
          </div>
        </header>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-3">
            <RegimeTimelineChart history={asset.history} />
            <EquityChart history={asset.history} />
            <RegimeProbStack history={asset.history} />
          </div>

          <aside className="space-y-3">
            <div className="panel p-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
                  CURRENT REGIME PROBABILITIES
                </div>
                <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent-green" />
                  <span className="text-accent-green">LIVE · {asset.as_of}</span>
                </div>
              </div>
              <div className="space-y-2">
                {(() => {
                  const probs = asset.current_regime.probs;
                  return probs.map((p, i) => {
                    const c = REGIME_COLORS[i] ?? "#a3a3a3";
                    const w = p === null ? 0 : Math.max(0, Math.min(1, p));
                    const isActive = i === activeLabel;
                    return (
                      <div key={i}>
                        <div className="flex items-center justify-between font-mono text-[11px]">
                          <span className="flex items-center gap-1.5" style={{ color: c }}>
                            <span>
                              {["Bull", "Neutral", "Bear"][i]}
                            </span>
                            {isActive && (
                              <span
                                className="rounded-sm px-1 py-px text-[8px] font-bold uppercase tracking-wider"
                                style={{
                                  backgroundColor: c,
                                  color: "#0a0e14",
                                }}
                                title="Current market regime"
                              >
                                ★ ACTIVE
                              </span>
                            )}
                          </span>
                          <span className="text-ink-muted">{fmtPct(p)}</span>
                        </div>
                        <div className="mt-0.5 h-1.5 w-full rounded bg-bg-ring">
                          <div
                            className="h-full rounded"
                            style={{
                              width: `${w * 100}%`,
                              backgroundColor: c,
                              opacity: isActive ? 1.0 : 0.55,
                            }}
                          />
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            </div>

            <div className="panel p-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
                  GMM + HMM · UNSUPERVISED REGIME
                </div>
                <div className="font-mono text-[10px] uppercase tracking-wider">
                  <span style={{ color: REGIME_COLORS[asset.current_gmm.label] ?? "#a3a3a3" }}>
                    {asset.current_gmm.name}
                  </span>
                  {asset.current_gmm.label === activeLabel ? (
                    <span className="ml-2 text-accent-green">✓ AGREES</span>
                  ) : (
                    <span className="ml-2 text-amber-400">⚠ DIVERGES</span>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                {asset.current_gmm.probs.map((p, i) => {
                  const c = REGIME_COLORS[i] ?? "#a3a3a3";
                  const w = p === null ? 0 : Math.max(0, Math.min(1, p));
                  const isActive = i === asset.current_gmm.label;
                  return (
                    <div key={i}>
                      <div className="flex items-center justify-between font-mono text-[11px]">
                        <span style={{ color: c }}>
                          {["Bull", "Neutral", "Bear"][i]}
                        </span>
                        <span className="text-ink-muted">{fmtPct(p)}</span>
                      </div>
                      <div className="mt-0.5 h-1.5 w-full rounded bg-bg-ring">
                        <div
                          className="h-full rounded"
                          style={{
                            width: `${w * 100}%`,
                            backgroundColor: c,
                            opacity: isActive ? 1.0 : 0.55,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-3 text-[10px] leading-relaxed text-ink-dim">
                3-state Gaussian HMM fit on (log-return, EWMA-vol). Unsupervised — states ranked by ascending variance. Agreement with the rule baseline = high confidence.
              </p>
            </div>

            <TvtpBand history={asset.history} />

            <TransitionHeatmap matrix={asset.transition_matrix} />

            <div className="panel p-4 text-xs">
              <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-ink-dim">
                CPCV BACKTEST · 10Y · 45 PATHS
              </div>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 font-mono">
                <dt className="text-ink-dim">Sharpe p05</dt>
                <dd className="text-right text-ink">{fmtSigned(asset.stats.sharpe_p05)}</dd>
                <dt className="text-ink-dim">Sharpe p50</dt>
                <dd className="text-right text-accent-lblue">{fmtSigned(asset.stats.sharpe_p50)}</dd>
                <dt className="text-ink-dim">Sharpe p95</dt>
                <dd className="text-right text-ink">{fmtSigned(asset.stats.sharpe_p95)}</dd>
                <dt className="text-ink-dim">Max DD p50</dt>
                <dd className="text-right text-accent-red">{fmtPct(asset.stats.max_dd_p50)}</dd>
                <dt className="text-ink-dim">DSR Sharpe</dt>
                <dd className="text-right text-ink">{fmtSigned(asset.stats.dsr_sharpe)}</dd>
              </dl>
              <p className="mt-3 text-[10px] leading-relaxed text-ink-dim">
                TVTP-MSAR position maps state 0 (low-vol) → +100% and state 1 (high-vol) → −30% defense. Strategy is the universe-wide champion per Regime_v2's CPCV harness.
              </p>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

function Stat({
  label,
  value,
  accent,
  hint,
}: {
  label: string;
  value: string;
  accent: string;
  hint?: string;
}) {
  return (
    <div className="panel min-w-[120px] px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
        {label}
      </div>
      <div
        className="font-mono text-lg font-bold leading-tight"
        style={{ color: accent }}
      >
        {value}
      </div>
      {hint && (
        <div className="mt-0.5 font-mono text-[9px] uppercase text-ink-dim">
          {hint}
        </div>
      )}
    </div>
  );
}
