"""
decision_logger.py
------------------
Decision/Logging — Decision Logger.

Writes a DecisionRecord (state snapshot + candidates + selected action)
to runs/decision_logs/{experiment_id}_{iteration}.json after every cycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from stratml.core.schemas import (
    ActionDecision,
    CandidateAction,
    DecisionRecord,
    StateObject,
)

_LOG_DIR = Path("runs/decision_logs")


def log(
    state: StateObject,
    candidates: list[CandidateAction],
    decision: ActionDecision,
    coordinator_weights: dict[str, float] | None = None,
    ranked_candidates: list[dict] | None = None,
    execution_result: dict | None = None,
    evaluator_result: dict | None = None,
    next_state_id: str | None = None,
) -> Path:
    """Persist DecisionRecord to disk. Returns the written file path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = DecisionRecord(
        experiment_id=state.meta.experiment_id,
        iteration=state.meta.iteration,
        timestamp=datetime.now(timezone.utc).isoformat(),
        state_snapshot=state,
        candidate_actions=candidates,
        selected_action=decision,
        coordinator_weights=coordinator_weights,
        ranked_candidates=ranked_candidates,
        execution_result=execution_result,
        evaluator_result=evaluator_result,
        next_state_id=next_state_id,
    )

    filename = f"{state.meta.experiment_id}_{state.meta.iteration:04d}.json"
    path = _LOG_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(record.model_dump_json(indent=2))

    return path


def update_outcome(
    experiment_id: str,
    iteration: int,
    execution_result: dict | None = None,
    evaluator_result: dict | None = None,
    next_state_id: str | None = None,
) -> Path | None:
    """Update an existing DecisionRecord on disk with execution outcome, evaluator verdict, and next state ID."""
    filename = f"{experiment_id}_{iteration:04d}.json"
    path = _LOG_DIR / filename
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if execution_result is not None:
            data["execution_result"] = execution_result
        if evaluator_result is not None:
            data["evaluator_result"] = evaluator_result
        if next_state_id is not None:
            data["next_state_id"] = next_state_id
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None
