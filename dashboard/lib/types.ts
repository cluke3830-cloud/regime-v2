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

// Phase 2 — categorical confidence derived from the fusion posterior.
// level rules: high if top > 0.65 AND second < 0.25; medium if top > 0.50;
// low otherwise. score = 1 - normalized_entropy (closer to 1 = more peaked).
export type ConfidenceLevel = "high" | "medium" | "low";

export interface CurrentConfidence {
  level: ConfidenceLevel;
  score: number;
  top_regime: string | null;
  top_prob: number | null;
  second_regime: string | null;
  second_prob: number | null;
  margin: number | null;
  reason: string;
}

// Phase 2 — cross-model agreement promoted from dashboard layer
// (confirmedActiveLabel) to first-class API contract. Counts how many of
// rule/gmm/tvtp match fusion's argmax. Useful for non-dashboard clients
// (alerts, third-party integrations) that need to branch on consensus
// without re-implementing the rules.
export type ConsensusLevel = "unanimous" | "strong" | "split" | "divided";

export interface ModelConsensus {
  models: {
    rule: number;
    gmm: number;
    tvtp: number | null;
    fusion: number;
  };
  model_names: {
    rule: string;
    gmm: string;
    tvtp: string;
    fusion: string;
  };
  agreement_count: number;
  agreement_pct: number;
  dissenters: string[];
  level: ConsensusLevel;
}

// Phase 2 — fusion meta surfaced explicitly so dashboard can read fusion.entropy
// without poking into the payload via untyped lookups. Optional for back-compat.
export interface FusionMeta {
  label: number;
  name: string;
  probs: (number | null)[];
  entropy: number | null;
}

// Phase 3 — heuristic transition-risk signal. Composed from three model-free
// sub-signals on the fusion posterior history:
//   margin_compression       — top-second prob margin shrinking
//   regime_persistence       — current regime overstaying vs typical p75 duration
//   second_prob_acceleration — second-best regime's prob rising fast
// Level rules: high if combined >= 0.65 OR any sub-score >= 0.75;
// medium if combined >= 0.35; low otherwise.
export type TransitionRiskLevel = "high" | "medium" | "low";
export type TransitionRiskSource = "fusion" | "rule_baseline";

export interface TransitionRiskComponentMargin {
  score: number;
  compression_pct: number | null;
  margin_now: number | null;
  margin_lookback: number | null;
}

export interface TransitionRiskComponentPersistence {
  score: number;
  current_streak: number;
  typical_p75: number | null;
  persistence_percentile: number;
}

export interface TransitionRiskComponentAcceleration {
  score: number;
  second_regime: string;
  delta: number | null;
  second_prob_now: number | null;
}

export interface TransitionRisk {
  level: TransitionRiskLevel;
  score: number;
  current_regime: string;
  days_in_regime: number;
  typical_p75_duration: number | null;
  expected_remaining_days: number;
  top_alternative_regime: string;
  components: {
    margin_compression: TransitionRiskComponentMargin;
    regime_persistence: TransitionRiskComponentPersistence;
    second_prob_acceleration: TransitionRiskComponentAcceleration;
  };
  reasons: string[];
  source_model: TransitionRiskSource;
}

export interface AssetPayload {
  ticker: string;
  name: string;
  as_of: string;
  current_close: number | null;
  current_regime: RegimeMeta;
  current_tvtp: TvtpMeta;
  current_gmm: GmmMeta;
  // Optional for backward-compat: payloads built before Phase 2 won't have
  // these. Consumers should null-check before reading.
  current_fusion?: FusionMeta;
  current_confidence?: CurrentConfidence;
  model_consensus?: ModelConsensus;
  transition_risk?: TransitionRisk;
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

// Trust the rule_baseline hard label by default. It has hysteresis built in
// and is the canonical bear filter — GMM-HMM and TVTP are unreliable on some
// assets (GMM was stuck at Bull for BTC's entire Nov 2025 bear, etc.).
//
// Override only when the soft posteriors DECISIVELY contradict the label:
// the labeled regime's probability has dropped below STALE_THRESHOLD *and*
// some other regime's probability has risen above DOMINANT_THRESHOLD. This
// auto-releases stale bears (BTC May 2026: rule=Bear, p2=0.16, p0=0.64 →
// override → Bull) without losing real bears (BTC Nov 2025: rule=Bear,
// p2≈0.25 → no override → Bear).
const STALE_THRESHOLD = 0.25;
const DOMINANT_THRESHOLD = 0.55;

export function confirmedActiveLabel(bar: {
  p0: number | null;
  p1: number | null;
  p2: number | null;
  label: number;
}): number {
  const probs: (number | null)[] = [bar.p0, bar.p1, bar.p2];
  const labelP = probs[bar.label];
  if (labelP === null || labelP === undefined) return bar.label;
  if (labelP >= STALE_THRESHOLD) return bar.label;
  let bestOther = -Infinity;
  let bestOtherIdx = bar.label;
  for (let c = 0; c < 3; c++) {
    if (c === bar.label) continue;
    const v = probs[c];
    if (v !== null && v !== undefined && v > bestOther) {
      bestOther = v;
      bestOtherIdx = c;
    }
  }
  return bestOther > DOMINANT_THRESHOLD ? bestOtherIdx : bar.label;
}

// Same rule applied to the asset payload's "current" fields.
export function confirmedActiveLabelFromAsset(asset: {
  current_regime: { probs: (number | null)[]; label: number };
}): number {
  return confirmedActiveLabel({
    p0: asset.current_regime.probs[0] ?? null,
    p1: asset.current_regime.probs[1] ?? null,
    p2: asset.current_regime.probs[2] ?? null,
    label: asset.current_regime.label,
  });
}
