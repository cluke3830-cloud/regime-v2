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
