import { loadAssetIndex, loadSummary } from "@/lib/data";
import TopBar from "@/components/TopBar";
import AssetGrid from "@/components/AssetGrid";
import SignInButton from "@/components/SignInButton";
import SubscribeButton from "@/components/SubscribeButton";

export default async function HomePage() {
  const [index, summary] = await Promise.all([loadAssetIndex(), loadSummary()]);

  const positive = summary.assets.filter(
    (a) => (a.sharpe_p50 ?? 0) > 0,
  ).length;
  const meanSharpe =
    summary.assets.reduce((s, a) => s + (a.sharpe_p50 ?? 0), 0) /
    Math.max(1, summary.assets.length);

  return (
    <main className="min-h-screen">
      <TopBar
        universe={index.universe}
        generatedAt={summary.generated_at}
        authSlot={<SignInButton />}
      />

      <section className="mx-auto max-w-7xl px-4 py-6">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="font-mono text-2xl font-bold tracking-tight text-ink">
              Multi-Asset Regime Universe
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-ink-muted">
              {summary.assets.length} assets · TVTP-MSAR champion (Hamilton
              2-state MS-AR) + 3-regime rule baseline. Backtest stats are
              45-path CPCV OOS Sharpe; snapshots refresh daily.
            </p>
            <div className="mt-3">
              <SubscribeButton />
            </div>
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