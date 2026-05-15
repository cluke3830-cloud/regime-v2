"use client";

import { useMemo, useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";

interface UniverseItem {
  ticker: string;
  safe: string;
  name: string;
}

// Valid US stock ticker pattern
const TICKER_RE = /^[A-Z0-9.\-^]{1,20}$/i;

export default function TopBar({
  universe,
  generatedAt,
}: {
  universe: UniverseItem[];
  generatedAt: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return universe;
    return universe.filter(
      (u) =>
        u.ticker.toLowerCase().includes(q) || u.name.toLowerCase().includes(q),
    );
  }, [universe, query]);

  // Show the "Analyze any US stock" row when query looks like a ticker and
  // isn't already an exact match in the universe
  const trimmed = query.trim().toUpperCase();
  const isValidTicker = TICKER_RE.test(trimmed);
  const exactMatch = universe.find((u) => u.ticker === trimmed);
  const showAnalyzeRow = isValidTicker && !exactMatch && trimmed.length >= 1;

  // Total dropdown rows including the optional analyze row
  const totalRows = matches.length + (showAnalyzeRow ? 1 : 0);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  function go(path: string) {
    setOpen(false);
    setQuery("");
    router.push(path);
  }

  function goTicker(safe: string) {
    go(`/asset/${safe}`);
  }

  function analyzeArbitrary() {
    go(`/asset/${trimmed}`);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusIdx((i) => Math.min(totalRows - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusIdx((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (focusIdx < matches.length) {
        const target = matches[focusIdx];
        if (target) goTicker(target.safe);
      } else if (showAnalyzeRow) {
        analyzeArbitrary();
      } else if (trimmed && isValidTicker) {
        // Enter on empty match list — navigate to the typed ticker
        analyzeArbitrary();
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <header className="relative z-30 border-b border-bg-ring bg-bg-panel/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
        <Link href="/" className="flex items-center gap-3 shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded bg-accent-dblue text-xs font-bold">
            R
          </div>
          <div className="hidden sm:block">
            <div className="text-sm font-bold tracking-tight glow-blue">
              REGIME<span className="text-accent-lblue">_</span>V2
            </div>
            <div className="font-mono text-[10px] text-ink-dim">
              MULTI-ASSET LIVE REGIME
            </div>
          </div>
        </Link>

        <div ref={wrapRef} className="relative flex-1 max-w-xl">
          <input
            type="text"
            placeholder="Search or type any US stock ticker (AAPL, NVDA…)  ⌘K"
            value={query}
            onFocus={() => setOpen(true)}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
              setFocusIdx(0);
            }}
            onKeyDown={onKeyDown}
            className="w-full rounded border border-bg-ring bg-bg-card px-3 py-2 font-mono text-sm
                       text-ink placeholder:text-ink-dim
                       focus:border-accent-lblue focus:outline-none focus:ring-1 focus:ring-accent-lblue"
          />
          {open && (
            <div className="absolute left-0 right-0 top-full mt-1 max-h-96 overflow-y-auto
                            rounded border border-bg-ring bg-bg-card shadow-2xl">
              {matches.length === 0 && !showAnalyzeRow && (
                <div className="px-3 py-2 text-sm text-ink-muted">No match.</div>
              )}

              {matches.map((m, i) => (
                <button
                  key={m.safe}
                  onClick={() => goTicker(m.safe)}
                  onMouseEnter={() => setFocusIdx(i)}
                  className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm font-mono ${
                    i === focusIdx
                      ? "bg-accent-dblue/40 text-ink"
                      : "text-ink hover:bg-bg-ring/50"
                  }`}
                >
                  <span className="font-bold text-accent-lblue">{m.ticker}</span>
                  <span className="text-xs text-ink-muted">{m.name}</span>
                </button>
              ))}

              {/* On-demand analyze row for any arbitrary US stock ticker */}
              {showAnalyzeRow && (
                <button
                  onClick={analyzeArbitrary}
                  onMouseEnter={() => setFocusIdx(matches.length)}
                  className={`flex w-full items-center justify-between border-t border-bg-ring px-3 py-2.5 text-left font-mono text-sm ${
                    focusIdx === matches.length
                      ? "bg-accent-dblue/40 text-ink"
                      : "text-ink hover:bg-bg-ring/50"
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <span className="text-accent-green">↗ Analyze</span>
                    <span className="font-bold text-accent-lblue">{trimmed}</span>
                  </span>
                  <span className="text-[10px] uppercase tracking-wider text-ink-dim">
                    any US stock · ~5–15s
                  </span>
                </button>
              )}
            </div>
          )}
        </div>

        <nav className="hidden md:flex items-center gap-2 text-xs font-mono">
          <Link
            href="/"
            className={`rounded px-2 py-1 ${
              pathname === "/" ? "bg-accent-dblue text-ink" : "text-ink-muted hover:text-ink"
            }`}
          >
            UNIVERSE
          </Link>
          <span className="text-ink-dim">•</span>
          <span className="text-ink-dim">DATA AS OF</span>
          <span className="text-ink">{generatedAt.slice(0, 10)}</span>
        </nav>
      </div>
    </header>
  );
}