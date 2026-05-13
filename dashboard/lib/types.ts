export type RegimeLabel = 0 | 1 | 2;

export interface RegimeMeta {
  label: RegimeLabel;
  name: string;
  alloc: number;
  color: string;
  probs: (number | null)[];
}

export interface TvtpMeta {
  p_low_vol: number;
  p_high_vol: number;
  position: number;
  state: string;
}

export interface HistoryBar {
  date: string;
  close: number | null;
  label: number;
  regime: string;
  alloc: number | null;
  p0: number | null;
  p1: number | null;
  p2: number | null;
  gmm_label: number;
  gmm_p0: number | null;
  gmm_p1: number | null;
  gmm_p2: number | null;
  tvtp_low: number | null;
  tvtp_high: number | null;
  tvtp_pos: number | null;
  eq_tvtp: number | null;
  eq_rule: number | null;
  eq_bh: number | null;
}

export interface GmmMeta {
  label: number;
  name: string;
  probs: (number | null)[];
}

export interface AssetPayload {
  ticker: string;
  name: string;
  as_of: string;
  current_close: number | null;
  current_regime: RegimeMeta;
  current_tvtp: TvtpMeta;
  current_gmm: GmmMeta;
  stats: {
    sharpe_p05?: number;
    sharpe_p50?: number;
    sharpe_p95?: number;
    max_dd_p50?: number;
    dsr_sharpe?: number;
  };
  transition_matrix: number[][];
  history: HistoryBar[];
}

export interface AssetSummaryItem {
  ticker: string;
  safe: string;
  name: string;
  as_of: string;
  close: number | null;
  regime: RegimeMeta;
  tvtp: TvtpMeta;
  sharpe_p50: number | null;
  max_dd_p50: number | null;
}

export interface SummaryPayload {
  generated_at: string;
  n_assets: number;
  regime_names: Record<string, string>;
  regime_alloc: Record<string, number>;
  regime_colors: Record<string, string>;
  assets: AssetSummaryItem[];
}

export interface AssetIndex {
  generated_at: string;
  universe: { ticker: string; safe: string; name: string }[];
}

export const REGIME_COLORS: Record<number, string> = {
  0: "#22c55e",  // Bull    — green
  1: "#a3a3a3",  // Neutral — grey
  2: "#ef4444",  // Bear    — red
};

export const REGIME_NAMES: Record<number, string> = {
  0: "Bull",
  1: "Neutral",
  2: "Bear",
};

// Canonical positions per regime — matches the rule-baseline allocation
// used by the strategy: Bull long, Neutral flat, Bear half-short.
export const REGIME_ALLOC: Record<number, number> = {
  0: 1.0,
  1: 0.0,
  2: -0.5,
};

// Argmax of the soft posteriors. The rule-baseline's hard `label` can lag
// or disagree with its own probabilities (hysteresis / thresholding), which
// produces dashboards where the badge says BEAR while P(Bull)=64%. The
// display layer everywhere uses this helper so the active regime always
// matches what the user sees in the probability stack.
export function activeRegimeFromProbs(
  probs: (number | null | undefined)[],
  fallback: number,
): number {
  let best = -1;
  let bestV = -Infinity;
  for (let i = 0; i < probs.length; i++) {
    const v = probs[i];
    if (v !== null && v !== undefined && v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best === -1 ? fallback : best;
}

// argmax of (p0, p1, p2) on a single history bar, with fallback to the
// stored hard label if probabilities are missing.
export function activeLabelOfBar(bar: {
  p0: number | null;
  p1: number | null;
  p2: number | null;
  label: number;
}): number {
  return activeRegimeFromProbs([bar.p0, bar.p1, bar.p2], bar.label);
}

// TVTP is a 2-state model: low-vol (Bull-ish) vs high-vol (Bear-ish).
// p_high > 0.5 → vote Bear; p_low > 0.5 → vote Bull; otherwise → Neutral.
function tvtpVote(tvtpLow: number | null, tvtpHigh: number | null): number {
  if (tvtpHigh !== null && tvtpHigh > 0.5) return 2;
  if (tvtpLow !== null && tvtpLow > 0.5) return 0;
  return 1;
}

// Two-of-three confirmation: returns the regime that gets ≥2 votes across
// (rule_argmax, gmm_label, tvtp_2state). Falls back to rule argmax when all
// three disagree. Catches bear (or bull) shifts ~1 day earlier than the
// rule baseline alone when two of the three models flip together.
export function confirmedActiveLabel(bar: {
  p0: number | null;
  p1: number | null;
  p2: number | null;
  gmm_label: number;
  tvtp_low: number | null;
  tvtp_high: number | null;
  label: number;
}): number {
  const vRule = activeRegimeFromProbs([bar.p0, bar.p1, bar.p2], bar.label);
  const vGmm = bar.gmm_label;
  const vTvtp = tvtpVote(bar.tvtp_low, bar.tvtp_high);
  const counts = [0, 0, 0];
  counts[vRule] += 1;
  counts[vGmm] += 1;
  counts[vTvtp] += 1;
  let best = vRule;
  let bestC = counts[vRule];
  for (let c = 0; c < 3; c++) {
    if (counts[c] >= 2 && counts[c] > bestC) {
      best = c;
      bestC = counts[c];
    }
  }
  return best;
}

// Same 2-of-3 confirmation, but using the asset payload's "current" fields
// (current_regime.probs, current_gmm.label, current_tvtp.{p_low_vol,p_high_vol}).
export function confirmedActiveLabelFromAsset(asset: {
  current_regime: { probs: (number | null)[]; label: number };
  current_gmm: { label: number };
  current_tvtp: { p_low_vol: number; p_high_vol: number };
}): number {
  const vRule = activeRegimeFromProbs(
    asset.current_regime.probs,
    asset.current_regime.label,
  );
  const vGmm = asset.current_gmm.label;
  const vTvtp = tvtpVote(asset.current_tvtp.p_low_vol, asset.current_tvtp.p_high_vol);
  const counts = [0, 0, 0];
  counts[vRule] += 1;
  counts[vGmm] += 1;
  counts[vTvtp] += 1;
  let best = vRule;
  let bestC = counts[vRule];
  for (let c = 0; c < 3; c++) {
    if (counts[c] >= 2 && counts[c] > bestC) {
      best = c;
      bestC = counts[c];
    }
  }
  return best;
}
