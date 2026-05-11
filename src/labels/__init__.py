"""Label engineering — triple-barrier and meta-labelling.

Phase 1 of the regime upgrade plan. The audit (§5.6.2) flagged the existing
LSTM forward-window teacher signal — derived from realised future returns,
volatility, and drawdown — as a labelling architecture that couples the
LSTM target to the same price process the rule features read from. Triple-
barrier labels (López de Prado AFML §3.2) replace it with a path-dependent,
event-driven label whose horizon is data-defined rather than fixed.
"""

from src.labels.triple_barrier import triple_barrier_labels

__all__ = ["triple_barrier_labels"]