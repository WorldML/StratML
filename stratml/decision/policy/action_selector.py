"""
action_selector.py
------------------
Decision/Policy — Action Selection Policy.

Picks the top-ranked action from the coordinator's output and
builds the final ActionDecision returned to Team A.

When the coordinator used the LLM path, reason.source is "learned" and
the coordinator's rationale is stored in reason.evidence["rationale"].
"""

from __future__ import annotations

from stratml.core.schemas import (
    ActionDecision,
    DecisionReason,
    PreprocessingConfig,
    StateObject,
)
import random
from pathlib import Path

from stratml.decision.agents.coordinator_agent import RankedAction

_EPSILON_LOW_DATA = 0.20
_EPSILON_HIGH_DATA = 0.05
_MIN_ROWS = 50


def _row_count() -> int:
    """Count filled observed_gain rows in the unified dataset."""
    try:
        import pandas as pd
        p = Path("runs/decision_logs/decision_dataset.csv")
        if not p.exists():
            return 0
        df = pd.read_csv(p)
        return int(df["observed_gain"].notna().sum())
    except Exception:
        return 0

_TREE_MODELS = {
    "RandomForestClassifier", "GradientBoostingClassifier",
    "ExtraTreesClassifier", "DecisionTreeClassifier",
}


def _build_preprocessing(state: StateObject, action_type: str = "", parameters: dict | None = None) -> PreprocessingConfig:
    parameters = parameters or {}
    imbalance = "oversample" if (state.dataset.imbalance_ratio or 1.0) > 2.0 else "none"
    missing = "median" if (state.dataset.missing_ratio or 0.0) > 0.1 else "mean"
    scaling = "none" if state.model.model_name in _TREE_MODELS else "standard"
    feature_selection = "none"

    if action_type in ("add_preprocessing", "apply_preprocessing"):
        strat = parameters.get("strategy")
        if strat in ("oversample", "undersample", "none"):
            imbalance = strat
        elif strat in ("standard", "minmax", "robust"):
            scaling = strat
        if "imbalance_strategy" in parameters:
            imbalance = parameters["imbalance_strategy"]
        if "scaling" in parameters:
            scaling = parameters["scaling"]
        if "missing_value_strategy" in parameters:
            missing = parameters["missing_value_strategy"]
        if "feature_selection" in parameters:
            feature_selection = parameters["feature_selection"]

    return PreprocessingConfig(
        missing_value_strategy=missing,
        scaling=scaling,
        encoding="onehot",
        imbalance_strategy=imbalance,
        feature_selection=feature_selection,
    )


def select(
    state: StateObject,
    ranked: list[RankedAction],
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
    decision_source: Optional[str] = None,
) -> ActionDecision:
    """Pick an action using epsilon-greedy exploration, then return an ActionDecision."""
    local_rng = rng if rng is not None else (random.Random(seed) if seed is not None else random)
    if state.resources.budget_exhausted:
        best = next((r for r in ranked if r.action_type == "terminate"), ranked[0])
        effective_source = "rule"
    else:
        epsilon = _EPSILON_LOW_DATA if _row_count() < _MIN_ROWS else _EPSILON_HIGH_DATA
        non_terminate = [r for r in ranked if r.action_type != "terminate"]
        if non_terminate:
            if decision_source == "llm":
                best = non_terminate[0]
                effective_source = "llm"
            elif local_rng.random() < epsilon:
                best = local_rng.choice(non_terminate)
                effective_source = "fallback"
            else:
                best = non_terminate[0]
                effective_source = decision_source or ("learned" if getattr(best, "rationale", "") else "rule")
        else:
            best = ranked[0]
            effective_source = decision_source or ("learned" if getattr(best, "rationale", "") else "rule")

    trigger = _infer_trigger(state)
    evidence = _build_evidence(state)

    if best.rationale:
        evidence["rationale"] = best.rationale

    return ActionDecision(
        experiment_id=state.meta.experiment_id,
        iteration=state.meta.iteration,
        action_type=best.action_type,
        parameters=best.parameters,
        preprocessing=_build_preprocessing(state, best.action_type, best.parameters),
        expected_gain=best.predicted_gain,
        expected_cost=best.predicted_cost,
        confidence=best.confidence,
        agent_scores=best.agent_scores,
        reason=DecisionReason(
            trigger=trigger,
            evidence=evidence,
            source=effective_source,
        ),
    )


def _infer_trigger(state: StateObject) -> str:
    sig = state.signals
    if state.meta.iteration == 0:
        return "bootstrap"
    if sig.converged != "none" and sig.well_fitted != "none":
        return "convergence"
    if sig.underfitting != "none":
        return "underfitting"
    if sig.overfitting != "none":
        return "overfitting"
    if sig.stagnating != "none" or sig.plateau_detected != "none":
        return "stagnation"
    if sig.diverging != "none":
        return "divergence"
    if sig.diminishing_returns != "none":
        return "diminishing_returns"
    return "exploration"


def _build_evidence(state: StateObject) -> dict:
    sig = state.signals
    t = state.trajectory
    return {
        "primary_metric": state.metrics.primary,
        "best_score": t.best_score,
        "slope": t.slope,
        "steps_since_improvement": t.steps_since_improvement,
        "underfitting": sig.underfitting != "none",
        "overfitting": sig.overfitting != "none",
        "converged": sig.converged != "none",
        "stagnating": sig.stagnating != "none",
        "remaining_budget": state.resources.remaining_budget,
    }
