"""Regime modelling — learned-classifier replacements for the rule layer.

Phase 2+ of the regime upgrade roadmap. Replaces the 540 LOC of hand-tuned
basis-function-and-weight scoring at regime_dashboard.py:986-1198 with
triple-barrier-labelled learned models, calibrated via the Phase 1 CPCV
harness.
"""

from src.regime.meta_stacker import (
    MetaStacker,
    make_equal_weight_stacked_strategy,
    make_ridge_stacked_strategy,
)
from src.regime.regime_xgboost import (
    RegimeXGBoost,
    compute_sample_weights,
    make_regime_xgboost_strategy,
)
from src.regime.rule_baseline import (
    REGIME_ALLOC,
    REGIME_NAMES,
    compute_rule_regime_sequence,
    rule_baseline_strategy,
)
from src.regime.conformal import (
    AdaptiveConformal,
    make_conformal_calibrated_strategy,
    regime_xgboost_proba_fn,
)
from src.regime.patchtst import (
    DeepEnsembleTransformer,
    TransformerRegimeClassifier,
    build_sequences,
    make_patchtst_strategy,
)
from src.regime.transition_detector import (
    TransitionDetector,
    evaluate_detector_metrics,
    make_transition_gated_strategy,
)
from src.regime.xgb_tuning import (
    DEFAULT_PARAM_GRID_FULL,
    DEFAULT_PARAM_GRID_SMALL,
    make_tuned_regime_xgboost_strategy,
    tune_xgb_hparams,
)

__all__ = [
    "RegimeXGBoost",
    "compute_sample_weights",
    "make_regime_xgboost_strategy",
    "tune_xgb_hparams",
    "make_tuned_regime_xgboost_strategy",
    "DEFAULT_PARAM_GRID_SMALL",
    "DEFAULT_PARAM_GRID_FULL",
    "REGIME_ALLOC",
    "REGIME_NAMES",
    "compute_rule_regime_sequence",
    "rule_baseline_strategy",
    "MetaStacker",
    "make_equal_weight_stacked_strategy",
    "make_ridge_stacked_strategy",
    "TransitionDetector",
    "evaluate_detector_metrics",
    "make_transition_gated_strategy",
    "TransformerRegimeClassifier",
    "DeepEnsembleTransformer",
    "build_sequences",
    "make_patchtst_strategy",
    "AdaptiveConformal",
    "make_conformal_calibrated_strategy",
    "regime_xgboost_proba_fn",
]