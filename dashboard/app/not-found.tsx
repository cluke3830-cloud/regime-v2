import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-bg p-8">
      <div className="panel max-w-md p-8 text-center">
        <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
          404 · NO ROUTE
        </div>
        <h1 className="mt-2 font-mono text-2xl font-bold text-ink">
          Asset not in universe
        </h1>
        <p className="mt-2 text-sm text-ink-muted">
          Regime_v2 ships SPY, QQQ, DIA, IWM, EFA, EEM, GLD, TLT, BTC-USD,
          JPY=X. Pick one from the home grid.
        </p>
        <Link
          href="/"
          className="mt-5 inline-block rounded border border-accent-lblue px-4 py-1.5 font-mono text-xs uppercase tracking-wider text-accent-lblue hover:bg-accent-dblue hover:text-ink"
        >
          ← BACK TO UNIVERSE
        </Link>
      </div>
    </main>
  );
}
