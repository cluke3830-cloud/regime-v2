import { REGIME_COLORS } from "@/lib/types";

export default function RegimeBadge({
  label,
  name,
  alloc,
  size = "md",
}: {
  label: number;
  name: string;
  alloc?: number;
  size?: "sm" | "md" | "lg";
}) {
  const color = REGIME_COLORS[label] ?? "#a3a3a3";
  const sizeCls =
    size === "sm"
      ? "text-[10px] px-1.5 py-0.5"
      : size === "lg"
      ? "text-sm px-3 py-1.5"
      : "text-xs px-2 py-1";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-sm font-mono uppercase tracking-wider ${sizeCls}`}
      style={{
        backgroundColor: `${color}1a`,
        color,
        border: `1px solid ${color}66`,
      }}
    >
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      <span>{name}</span>
      {alloc !== undefined && (
        <span className="text-ink-muted">
          {alloc >= 0 ? "+" : ""}
          {(alloc * 100).toFixed(0)}%
        </span>
      )}
    </span>
  );
}
