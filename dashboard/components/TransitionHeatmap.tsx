import { REGIME_COLORS, REGIME_NAMES } from "@/lib/types";

export default function TransitionHeatmap({ matrix }: { matrix: number[][] }) {
  const n = matrix.length;
  const max = Math.max(0.001, ...matrix.flat());
  return (
    <div className="panel p-3">
      <div className="mb-2">
        <div className="font-mono text-[10px] uppercase tracking-wider text-ink-dim">
          REGIME TRANSITION MATRIX
        </div>
        <div className="text-sm text-ink">
          Empirical P(next regime | current) — last ~3 years
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-xs">
          <thead>
            <tr>
              <th className="p-1 text-left text-ink-muted">from \ to</th>
              {Array.from({ length: n }).map((_, j) => (
                <th
                  key={j}
                  className="p-1 text-center"
                  style={{ color: REGIME_COLORS[j] }}
                >
                  {REGIME_NAMES[j]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="p-1 text-right" style={{ color: REGIME_COLORS[i] }}>
                  {REGIME_NAMES[i]}
                </td>
                {row.map((p, j) => {
                  const intensity = max > 0 ? p / max : 0;
                  return (
                    <td
                      key={j}
                      className="text-center"
                      style={{
                        backgroundColor: `rgba(46, 117, 182, ${Math.max(0.05, intensity * 0.85).toFixed(2)})`,
                        border: i === j ? `1px solid ${REGIME_COLORS[i]}` : "1px solid #1e2a3a",
                        padding: "10px 6px",
                      }}
                    >
                      <span className="text-ink">{(p * 100).toFixed(1)}%</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 font-mono text-[10px] text-ink-dim">
        Diagonal = persistence. Off-diagonal = transition rate. Darker blue = higher.
      </div>
    </div>
  );
}
