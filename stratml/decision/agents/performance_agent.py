"""
performance_agent.py
--------------------
Decision Council — Performance Agent.

LLM path: LangChain chain with structured output (llama-3.3-70b-versatile).
Fallback: rule-based lookup table when LLM call fails or GROQ_API_KEY is absent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel

from stratml.core.schemas import StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate
from stratml.decision.llm_control import is_llm_enabled

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_FITTING_PRIORITY: dict[str, dict[str, float]] = {
    "underfitting": {
        "switch_model": 0.85,
        "increase_model_capacity": 0.80,
        "modify_regularization": 0.40,
        "decrease_model_capacity": 0.10,
        "change_optimizer": 0.50,
        "terminate": 0.05,
    },
    "overfitting": {
        "switch_model": 0.60,
        "increase_model_capacity": 0.10,
        "modify_regularization": 0.85,
        "decrease_model_capacity": 0.75,
        "change_optimizer": 0.40,
        "terminate": 0.20,
    },
    "well_fitted": {
        "switch_model": 0.80,
        "increase_model_capacity": 0.60,
        "modify_regularization": 0.50,
        "decrease_model_capacity": 0.30,
        "change_optimizer": 0.40,
        "terminate": 0.10,
    },
    "default": {
        "switch_model": 0.60,
        "increase_model_capacity": 0.50,
        "modify_regularization": 0.50,
        "decrease_model_capacity": 0.40,
        "change_optimizer": 0.45,
        "terminate": 0.30,
    },
}


def _rule_score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    sig = state.signals
    if sig.underfitting != "none":
        table = _FITTING_PRIORITY["underfitting"]
    elif sig.overfitting != "none":
        table = _FITTING_PRIORITY["overfitting"]
    elif sig.well_fitted != "none":
        table = _FITTING_PRIORITY["well_fitted"]
    else:
        table = _FITTING_PRIORITY["default"]

    scores = {
        e.action_type: round(table.get(e.action_type, 0.5) + e.predicted_gain * 0.3, 4)
        for e in estimates
    }

    if sig.plateau_detected != "none" and "switch_model" in scores:
        boost = 0.3 if sig.plateau_detected == "strong" else 0.15
        scores["switch_model"] = round(scores["switch_model"] + boost, 4)
        if "modify_regularization" in scores:
            scores["modify_regularization"] = round(scores["modify_regularization"] - 0.2, 4)

    return scores


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a performance analyst. Your only concern is whether a candidate action "
    "will improve the model's predictive accuracy given the current fitting state and "
    "trajectory. Score each action 0.0-1.0 on expected performance gain."
)


class _Scores(BaseModel):
    scores: dict[str, float]


def _llm_score(state: StateObject, estimates: list[UncertaintyEstimate]) -> Optional[dict[str, float]]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        sig = state.signals
        traj = state.trajectory
        candidates = [
            {"action_type": e.action_type, "parameters": e.parameters, "predicted_gain": e.predicted_gain}
            for e in estimates
        ]
        human = (
            f"Fitting: underfitting={sig.underfitting}, overfitting={sig.overfitting}, "
            f"well_fitted={sig.well_fitted}, plateau={sig.plateau_detected}\n"
            f"Trajectory: slope={traj.slope:.4f}, steps_since_improvement={traj.steps_since_improvement}, "
            f"trend={traj.trend}, best_score={traj.best_score:.4f}\n"
            f"Model: {state.model.model_name}\n"
            f"Candidates: {candidates}\n\n"
            "Return JSON with key 'scores' mapping each action_type to a float 0.0-1.0."
        )
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(_Scores)
        result: _Scores = llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human)])
        return {k: round(max(0.0, min(v, 1.0)), 4) for k, v in result.scores.items()}
    except Exception as exc:
        log.warning("performance_agent LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    """Return performance_score per action_type."""
    if is_llm_enabled():
        result = _llm_score(state, estimates)
        if result is not None:
            return result
    return _rule_score(state, estimates)
