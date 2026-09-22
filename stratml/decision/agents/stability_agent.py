"""
stability_agent.py
------------------
Decision Council — Stability Agent.

LLM path: LangChain chain with structured output (llama-3.3-70b-versatile).
Fallback: rule-based stability table when LLM call fails or GROQ_API_KEY is absent.
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

_STABILITY_BASE: dict[str, float] = {
    "terminate": 0.95,
    "modify_regularization": 0.85,
    "decrease_model_capacity": 0.80,
    "change_optimizer": 0.70,
    "switch_model": 0.65,
    "increase_model_capacity": 0.35,
}


def _rule_score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    sig = state.signals
    instability_penalty = 0.0
    if sig.diverging != "none":
        instability_penalty += 0.3
    if sig.unstable_training != "none":
        instability_penalty += 0.2
    if sig.high_variance != "none":
        instability_penalty += 0.1

    scores: dict[str, float] = {}
    for e in estimates:
        base = _STABILITY_BASE.get(e.action_type, 0.5)
        risk = (1.0 - base) * instability_penalty
        raw = base - risk - e.variance * 0.2
        scores[e.action_type] = round(max(0.0, min(raw, 1.0)), 4)
    return scores


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a training stability analyst. Your only concern is whether a candidate action "
    "risks destabilizing training given current variance, divergence, and loss signals. "
    "Score each action 0.0-1.0 on stability (higher = safer)."
)


class _Scores(BaseModel):
    scores: dict[str, float]


def _llm_score(state: StateObject, estimates: list[UncertaintyEstimate]) -> Optional[dict[str, float]]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        sig = state.signals
        gen = state.generalization
        candidates = [
            {"action_type": e.action_type, "variance": e.variance}
            for e in estimates
        ]
        human = (
            f"Stability signals: diverging={sig.diverging}, unstable_training={sig.unstable_training}, "
            f"high_variance={sig.high_variance}\n"
            f"Generalization: train_loss={gen.train_loss:.4f}, val_loss={gen.validation_loss:.4f}, "
            f"gap={gen.gap:.4f}\n"
            f"Trajectory volatility={state.trajectory.volatility:.4f}\n"
            f"Candidates: {candidates}\n\n"
            "Return JSON with key 'scores' mapping each action_type to a float 0.0-1.0."
        )
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(_Scores)
        result: _Scores = llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human)])
        return {k: round(max(0.0, min(v, 1.0)), 4) for k, v in result.scores.items()}
    except Exception as exc:
        log.warning("stability_agent LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    """Return stability_score per action_type. Higher = more stable choice."""
    if is_llm_enabled():
        result = _llm_score(state, estimates)
        if result is not None:
            return result
    return _rule_score(state, estimates)
