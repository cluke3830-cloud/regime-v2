"use client";

import { useEffect, useState } from "react";
import AssetDetail from "@/components/AssetDetail";
import type { AssetPayload } from "@/lib/types";

function ComputingState({ ticker }: { ticker: string }) {
  const [dots, setDots] = useState(".");
  useEffect(() => {
    const id = setInterval(() => setDots((d) => (d.length >= 3 ? "." : d + ".")), 600);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="mx-auto max-w-7xl px-4 py-16 text-center">
      <div className="mb-4 font-mono text-4xl font-bold tracking-tight text-ink">
        {ticker}
      </div>
      <div className="mb-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent-green" />
      <p className="font-mono text-sm text-ink-muted">
        Computing regime{dots}
      </p>
      <p className="mt-2 font-mono text-[11px] text-ink-dim">
        Fetching data · fitting HMM + TVTP-MSAR · fusing models · ~5–15s
      </p>

      {/* Skeleton cards */}
      <div className="mt-10 grid grid-cols-1 gap-3 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-3">
          {[200, 180, 160].map((h) => (
            <div
              key={h}
              className="panel animate-pulse"
              style={{ height: h }}
            />
          ))}
        </div>
        <div className="space-y-3">
          {[120, 140, 100, 100].map((h, i) => (
            <div
              key={i}
              className="panel animate-pulse"
              style={{ height: h }}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function ErrorState({ ticker, error }: { ticker: string; error: string }) {
  return (
    <section className="mx-auto max-w-7xl px-4 py-16 text-center">
      <div className="mb-3 font-mono text-4xl font-bold tracking-tight text-accent-red">
        {ticker}
      </div>
      <p className="font-mono text-sm text-accent-red">Failed to compute regime</p>
      <p className="mt-2 font-mono text-[11px] text-ink-dim max-w-lg mx-auto">{error}</p>
      <p className="mt-4 font-mono text-[11px] text-ink-dim">
        Check that the ticker is valid and the API server is reachable.
      </p>
    </section>
  );
}

export default function DynamicAssetLoader({ ticker }: { ticker: string }) {
  const [asset, setAsset] = useState<AssetPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/regime/${encodeURIComponent(ticker)}`)
      .then((r) => {
        if (!r.ok)
          return r.json().then((j) => {
            throw new Error(j.error ?? `HTTP ${r.status}`);
          });
        return r.json();
      })
      .then((data: AssetPayload) => {
        if (!cancelled) setAsset(data);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  if (error) return <ErrorState ticker={ticker} error={error} />;
  if (!asset) return <ComputingState ticker={ticker} />;
  return <AssetDetail asset={asset} />;
}