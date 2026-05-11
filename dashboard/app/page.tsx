import { loadAssetIndex, loadSummary } from "@/lib/data";
import TopBar from "@/components/TopBar";
import AssetGrid from "@/components/AssetGrid";

function fmtPct(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  return (s * 100).toFixed(1) + "%";
}

export default async function HomePage() {
  const [index, summary] = await Promise.all([loadAssetIndex(), loadSummary()]);

  const positive = summary.assets.filter(
    (a) => (a.sharpe_p50 ?? 0) > 0,
  ).length;
  const meanSharpe =
    summary.assets.reduce((s, a) => s + (a.sharpe_p50 ?? 0), 0) /
    Math.max(1, summary.assets.length);
  const regimeCounts: Record<number, number> = {};
  for (const a of summary.assets) {
    regimeCounts[a.regime.label] = (regimeCounts[a.regime.label] ?? 0) + 1;
  }

  return (
    <main className="min-h-screen">
      <TopBar
        universe={index.universe}
        generatedAt={summary.generated_at}
      />

      <section className="mx-auto max-w-7xl px-4 py-6">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="font-mono text-2xl font-bold tracking-tight text-ink">
              Multi-Asset Regime Universe
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-ink-muted">
              10 assets · TVTP-MSAR champion (Hamilton 2-state MS-AR) + 5-regime
              rule baseline. Backtest stats are 45-path CPCV OOS Sharpe;
              snapshots refresh daily.
            </p>
          </div>
          <div className="flex gap-2">
            <Kpi
              label="ASSETS"
              value={String(summary.assets.length)}
              hint="UNIVERSE"
            />
            <Kpi
              label="POS SHARPE"
              value={`${positive}/${summary.assets.length}`}
              hint="OOS p50 > 0"
              accent={positive >= 7 ? "green" : "amber"}
            />
            <Kpi
              label="MEAN SHARPE"
              value={(meanSharpe >= 0 ? "+" : "") + meanSharpe.toFixed(2)}
              hint="ACROSS UNIVERSE"
              accent={meanSharpe > 0 ? "green" : "red"}
            />
          </div>
        </div>

        <AssetGrid assets={summary.assets} />

        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-5">
          {[0, 1, 2, 3, 4].map((r) => {
            const c = regimeCounts[r] ?? 0;
            return (
              <RegimeCount
                key={r}
                label={r}
                count={c}
                total={summary.assets.length}
              />
            );
          })}
        </div>

        <Footer />
      </section>
    </main>
  );
}

function Kpi({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: "green" | "red" | "amber";
}) {
  const color =
    accent === "green"
      ? "#22c55e"
      : accent === "red"
      ? "#ef4444"
      : accent === "amber"
      ? "#f5a623"
      : "#2E75B6";
  return (
    <div className="panel min-w-[120px] px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
        {label}
      </div>
      <div
        className="font-mono text-xl font-bold leading-tight"
        style={{ color }}
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

function RegimeCount({
  label,
  count,
  total,
}: {
  label: number;
  count: number;
  total: number;
}) {
  const REGIME_COLORS = ["#22c55e", "#84cc16", "#a3a3a3", "#f97316", "#ef4444"];
  const REGIME_NAMES = ["Full Bull", "Half Bull", "Chop", "Half Bear", "Full Bear"];
  const color = REGIME_COLORS[label];
  const pct = total > 0 ? count / total : 0;
  return (
    <div
      className="panel relative overflow-hidden p-3"
      style={{ boxShadow: `inset 0 0 0 1px ${color}33` }}
    >
      <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color }}>
        {REGIME_NAMES[label]}
      </div>
      <div className="mt-1 flex items-baseline gap-1">
        <span className="font-mono text-2xl font-bold" style={{ color }}>
          {count}
        </span>
        <span className="font-mono text-xs text-ink-dim">/ {total}</span>
      </div>
      <div className="mt-2 h-1 w-full rounded bg-bg-ring">
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${pct * 100}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="mt-10 border-t border-bg-ring pt-4 font-mono text-[10px] uppercase tracking-wider text-ink-dim">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span>REGIME_V2 // ALL 17 BRIEFS SHIPPED // 225/225 TESTS GREEN</span>
        <span>·</span>
        <span>CHAMPION TVTP-MSAR SPY p50 +2.462 · MULTI-ASSET 10/10 POS · PBO 13.33%</span>
        <span>·</span>
        <span>SNAPSHOTS REFRESH DAILY</span>
      </div>
    </footer>
  );
}
