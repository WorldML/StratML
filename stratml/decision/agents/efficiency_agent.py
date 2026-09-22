"""
efficiency_agent.py
-------------------
Decision Council — Efficiency Agent.

LLM path: LangChain chain with structured output (llama-3.3-70b-versatile).
Fallback: rule-based cost table when LLM call fails or GROQ_API_KEY is absent.
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

_ACTION_COST: dict[str, float] = {
    "terminate": 0.0,
    "modify_regularization": 0.2,
    "decrease_model_capacity": 0.3,
    "change_optimizer": 0.4,
    "increase_model_capacity": 0.6,
    "switch_model": 0.7,
}


def _rule_score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    r = state.resources
    budget_pressure = 0.0
    if r.remaining_budget is not None and state.constraints.max_iterations > 0:
        used_ratio = 1.0 - (r.remaining_budget / state.constraints.max_iterations)
        budget_pressure = max(0.0, min(used_ratio, 1.0))
    runtime_pressure = min(r.runtime / 300.0, 1.0)

    scores: dict[str, float] = {}
    for e in estimates:
        base_cost = _ACTION_COST.get(e.action_type, 0.5)
        penalty = base_cost * (0.5 * budget_pressure + 0.5 * runtime_pressure)
        raw = 1.0 - base_cost - penalty * 0.3 - e.predicted_cost * 0.2
        scores[e.action_type] = round(max(0.0, min(raw, 1.0)), 4)
    return scores


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a resource analyst. Your only concern is compute cost and budget consumption. "
    "Score each action 0.0-1.0 on efficiency (higher = cheaper relative to remaining budget)."
)


class _Scores(BaseModel):
    scores: dict[str, float]


def _llm_score(state: StateObject, estimates: list[UncertaintyEstimate]) -> Optional[dict[str, float]]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        r = state.resources
        candidates = [
            {"action_type": e.action_type, "predicted_cost": e.predicted_cost}
            for e in estimates
        ]
        human = (
            f"Resources: runtime={r.runtime:.1f}s, remaining_budget={r.remaining_budget}, "
            f"budget_exhausted={r.budget_exhausted}, max_iterations={state.constraints.max_iterations}\n"
            f"Candidates: {candidates}\n\n"
            "Return JSON with key 'scores' mapping each action_type to a float 0.0-1.0."
        )
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(_Scores)
        result: _Scores = llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human)])
        return {k: round(max(0.0, min(v, 1.0)), 4) for k, v in result.scores.items()}
    except Exception as exc:
        log.warning("efficiency_agent LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    """Return efficiency_score per action_type. Higher = more efficient."""
    if is_llm_enabled():
        result = _llm_score(state, estimates)
        if result is not None:
            return result
    return _rule_score(state, estimates)
