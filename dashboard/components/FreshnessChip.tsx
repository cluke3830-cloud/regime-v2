"use client";

import { useEffect, useState } from "react";

interface Freshness {
  level: "fresh" | "dated" | "stale";
  label: string;
  title: string;
}

function compute(modelRunAt: string): Freshness {
  const hoursAgo = (Date.now() - new Date(modelRunAt).getTime()) / 3_600_000;
  const label =
    hoursAgo < 1
      ? `${Math.round(hoursAgo * 60)}m ago`
      : `${Math.round(hoursAgo)}h ago`;
  const title = `Model ran ${label} (${new Date(modelRunAt).toLocaleString()}). Confidence typically degrades after 48 h.`;

  if (hoursAgo < 6) return { level: "fresh", label, title };
  if (hoursAgo < 24) return { level: "dated", label, title };
  return { level: "stale", label, title };
}

const STYLES: Record<Freshness["level"], string> = {
  fresh: "bg-green-900/40  text-green-400  border-green-700",
  dated: "bg-amber-900/40  text-amber-400  border-amber-700",
  stale: "bg-red-900/40    text-red-400    border-red-700",
};

export default function FreshnessChip({ modelRunAt }: { modelRunAt: string }) {
  const [f, setF] = useState<Freshness | null>(null);

  useEffect(() => {
    setF(compute(modelRunAt));
    const id = setInterval(() => setF(compute(modelRunAt)), 60_000);
    return () => clearInterval(id);
  }, [modelRunAt]);

  // Render nothing on the server (avoids hydration mismatch with live clock)
  if (!f) return null;

  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10px] ${STYLES[f.level]}`}
      title={f.title}
    >
      {f.label}
    </span>
  );
}
