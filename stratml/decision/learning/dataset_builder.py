"""
dataset_builder.py
------------------
Decision/Learning — Decision Dataset Builder.

Appends one row per decision cycle to:
  - outputs/<run_id>/decision_logs/decision_dataset.csv  (run-specific, set by DecisionEngine)
  - runs/decision_logs/decision_dataset.csv              (unified, always written)

observed_gain is backfilled by backfill_last_gain() when the next ExperimentResult arrives.
"""

from __future__ import annotations

import csv
from pathlib import Path

from stratml.core.schemas import CandidateAction, StateObject

# Run-specific path — overridden by DecisionEngine.__init__
_DATASET_PATH = Path("runs/decision_logs/decision_dataset.csv")

# Unified path — accumulates across all runs for value model training
_UNIFIED_PATH = Path("runs/decision_logs/decision_dataset.csv")

_COLUMNS = [
    "dataset_id",
    "run_id",
    "seed",
    "experiment_id",
    "iteration",
    "primary_metric",
    "best_score",
    "improvement_rate",
    "slope",
    "volatility",
    "steps_since_improvement",
    "trend",
    "underfitting",
    "overfitting",
    "well_fitted",
    "converged",
    "stagnating",
    "num_samples",
    "num_features",
    "missing_ratio",
    "runtime",
    "remaining_budget",
    "model_name",
    "complexity_hint",
    "action_type",
    "action_params",
    "predicted_gain",
    "observed_gain",       # backfilled by backfill_last_gain()
    "normalized_gain",     # observed_gain / (1 - best_score_at_decision), cross-dataset comparable
    "status",              # "pending", "completed", "failed", "incomplete"
]


def _append(path: Path, row: dict, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record(
    state: StateObject,
    action: CandidateAction,
    predicted_gain: float = 0.0,
    run_id: str | None = None,
    dataset_id: str | None = None,
    seed: int | None = None,
) -> None:
    """Append a row for the selected action. observed_gain left blank until backfilled."""
    eff_run_id = run_id or getattr(state.meta, "run_id", None) or "unknown_run"
    eff_dataset_id = (
        dataset_id
        or getattr(state.dataset, "dataset_id", None)
        or getattr(state.dataset, "dataset_name", None)
        or "unknown_dataset"
    )
    eff_seed = seed if seed is not None else getattr(state.meta, "seed", 42)

    row = {
        "dataset_id": eff_dataset_id,
        "run_id": eff_run_id,
        "seed": eff_seed,
        "experiment_id": state.meta.experiment_id,
        "iteration": state.meta.iteration,
        "primary_metric": state.objective.primary_metric,
        "best_score": state.trajectory.best_score,
        "improvement_rate": state.trajectory.improvement_rate,
        "slope": state.trajectory.slope,
        "volatility": state.trajectory.volatility,
        "steps_since_improvement": state.trajectory.steps_since_improvement,
        "trend": state.trajectory.trend,
        "underfitting": state.signals.underfitting,
        "overfitting": state.signals.overfitting,
        "well_fitted": state.signals.well_fitted,
        "converged": state.signals.converged,
        "stagnating": state.signals.stagnating,
        "num_samples": state.dataset.num_samples,
        "num_features": state.dataset.num_features,
        "missing_ratio": state.dataset.missing_ratio,
        "runtime": state.resources.runtime,
        "remaining_budget": state.resources.remaining_budget,
        "model_name": state.model.model_name or "none",
        "complexity_hint": state.model.complexity_hint or "none",
        "action_type": action.action_type,
        "action_params": str(action.parameters),
        "predicted_gain": predicted_gain,
        "observed_gain": "",
        "normalized_gain": "",
        "status": "pending",
    }

    _append(_DATASET_PATH, row, not _DATASET_PATH.exists())

    # Also write to unified path (skip if same as run-specific)
    if _UNIFIED_PATH != _DATASET_PATH:
        _append(_UNIFIED_PATH, row, not _UNIFIED_PATH.exists())


def backfill_last_gain(gain: float, status: str = "completed") -> None:
    """Update observed_gain and normalized_gain of the last unfilled row in both CSVs."""
    for path in {_DATASET_PATH, _UNIFIED_PATH}:
        if not path.exists():
            continue
        try:
            import csv as _csv
            rows = list(_csv.DictReader(path.open(encoding="utf-8")))
            if not rows:
                continue
            # Find last row with empty observed_gain
            for row in reversed(rows):
                if row.get("observed_gain", "").strip() == "":
                    row["observed_gain"] = str(round(gain, 6))
                    best = float(row.get("best_score") or 0.0)
                    headroom = 1.0 - best
                    row["normalized_gain"] = str(round(gain / headroom, 6)) if headroom > 1e-6 else str(round(gain, 6))
                    row["status"] = status
                    break
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("backfill_last_gain failed for %s: %s", path, exc)
