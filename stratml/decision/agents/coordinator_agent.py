"""
coordinator_agent.py
--------------------
Decision Council — Coordinator Agent.

LLM path: deliberates over the three agent scores, resolves disagreements,
produces a ranked list with a natural-language rationale per action.
Fallback: weighted sum (performance=0.50, efficiency=0.25, stability=0.25).

The rationale string is stored in DecisionReason.evidence["rationale"] downstream
(action_selector.py reads ranked[0]; the rationale travels via RankedAction.rationale).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from stratml.core.schemas import AgentScore, StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate

log = logging.getLogger(__name__)

_W_PERF_DEFAULT = 0.50
_W_EFF_DEFAULT  = 0.25
_W_STAB_DEFAULT = 0.25
_EMA_ALPHA = 0.2


def _load_agent_weights(log_paths: list[str | Path] | None = None) -> tuple[float, float, float]:
    """Compute per-agent EMA weights from evaluation_log.jsonl across all runs."""
    import glob
    from pathlib import Path
    if log_paths is not None:
        logs = [str(p) for p in log_paths]
    else:
        logs = glob.glob("outputs/*/decision_logs/evaluation_log.jsonl")
    if not logs:
        return _W_PERF_DEFAULT, _W_EFF_DEFAULT, _W_STAB_DEFAULT
    try:
        import json
        records = []
        for log_path in logs:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        if len(records) < 5:
            return _W_PERF_DEFAULT, _W_EFF_DEFAULT, _W_STAB_DEFAULT
        # Proxy: high validity + positive cf_impact => perf agent was right
        # low quality_risk => stability agent was right
        # low counterfactual_impact cost => efficiency agent was right
        w_p = w_e = w_s = 0.5
        for r in records:
            cf = r.get("counterfactual_impact", 0.0) or 0.0
            validity = r.get("decision_validity", 0.5) or 0.5
            risk = r.get("quality_risk", 0.5) or 0.5
            perf_right = 1.0 if (validity >= 0.5 and cf >= 0) else 0.0
            eff_right  = 1.0 if cf >= -0.02 else 0.0  # action didnt cost much
            stab_right = 1.0 if risk < 0.5 else 0.0
            w_p = (1 - _EMA_ALPHA) * w_p + _EMA_ALPHA * perf_right
            w_e = (1 - _EMA_ALPHA) * w_e + _EMA_ALPHA * eff_right
            w_s = (1 - _EMA_ALPHA) * w_s + _EMA_ALPHA * stab_right
        total = w_p + w_e + w_s or 1.0
        return round(w_p / total, 4), round(w_e / total, 4), round(w_s / total, 4)
    except Exception:
        return _W_PERF_DEFAULT, _W_EFF_DEFAULT, _W_STAB_DEFAULT


@dataclass
class RankedAction:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float
    confidence: float
    agent_scores: AgentScore
    final_score: float
    rationale: str = field(default="")


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_rank(
    state: StateObject,
    estimates: list[UncertaintyEstimate],
    perf_scores: dict[str, float],
    eff_scores: dict[str, float],
    stab_scores: dict[str, float],
    log_paths: list[str | Path] | None = None,
) -> list[RankedAction]:
    w_p, w_e, w_s = _load_agent_weights(log_paths=log_paths)
    ranked: list[RankedAction] = []
    for e in estimates:
        p  = perf_scores.get(e.action_type, 0.5)
        ef = eff_scores.get(e.action_type, 0.5)
        st = stab_scores.get(e.action_type, 0.5)
        final = round(w_p * p + w_e * ef + w_s * st, 4)
        ranked.append(RankedAction(
            action_type=e.action_type,
            parameters=e.parameters,
            predicted_gain=e.predicted_gain,
            predicted_cost=e.predicted_cost,
            confidence=e.confidence,
            agent_scores=AgentScore(performance=p, efficiency=ef, stability=st),
            final_score=final,
        ))
    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are the coordinator of a multi-agent ML experimentation system. "
    "Three specialist agents have scored candidate actions from their own perspectives: "
    "performance (accuracy gain), efficiency (compute cost), and stability (training risk). "
    "They may disagree. Your job is to reason about which action to select given the full "
    "experiment context, override agent scores if the situation warrants it (e.g., budget "
    "nearly exhausted means efficiency should dominate), and return a ranked list with a "
    "brief rationale for each action."
)


class _RankedItem(BaseModel):
    action_type: str
    final_score: float  # 0.0-1.0
    rationale: str


class _CoordinatorOutput(BaseModel):
    ranked: list[_RankedItem]


def _llm_rank(
    state: StateObject,
    estimates: list[UncertaintyEstimate],
    perf_scores: dict[str, float],
    eff_scores: dict[str, float],
    stab_scores: dict[str, float],
) -> Optional[list[RankedAction]]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        sig = state.signals
        traj = state.trajectory
        r = state.resources

        agent_scores_text = "\n".join(
            f"  {e.action_type}: perf={perf_scores.get(e.action_type, 0.5):.3f}, "
            f"eff={eff_scores.get(e.action_type, 0.5):.3f}, "
            f"stab={stab_scores.get(e.action_type, 0.5):.3f}"
            for e in estimates
        )
        human = (
            f"State summary:\n"
            f"  fitting: underfitting={sig.underfitting}, overfitting={sig.overfitting}, "
            f"well_fitted={sig.well_fitted}, plateau={sig.plateau_detected}\n"
            f"  trajectory: trend={traj.trend}, slope={traj.slope:.4f}, "
            f"steps_since_improvement={traj.steps_since_improvement}, best_score={traj.best_score:.4f}\n"
            f"  resources: remaining_budget={r.remaining_budget}, runtime={r.runtime:.1f}s, "
            f"budget_exhausted={r.budget_exhausted}\n"
            f"  model: {state.model.model_name}, iteration={state.meta.iteration}\n\n"
            f"Agent scores per candidate:\n{agent_scores_text}\n\n"
            "Return a ranked list of all candidates with a final_score (0.0-1.0) and a "
            "one-sentence rationale for each. Include every action_type listed above."
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(_CoordinatorOutput)
        output: _CoordinatorOutput = llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human)])

        # Build a lookup from LLM output
        llm_map = {item.action_type: item for item in output.ranked}

        ranked: list[RankedAction] = []
        for e in estimates:
            p  = perf_scores.get(e.action_type, 0.5)
            ef = eff_scores.get(e.action_type, 0.5)
            st = stab_scores.get(e.action_type, 0.5)
            llm_item = llm_map.get(e.action_type)
            final = round(max(0.0, min(llm_item.final_score, 1.0)), 4) if llm_item else round(_W_PERF * p + _W_EFF * ef + _W_STAB * st, 4)
            rationale = llm_item.rationale if llm_item else ""
            ranked.append(RankedAction(
                action_type=e.action_type,
                parameters=e.parameters,
                predicted_gain=e.predicted_gain,
                predicted_cost=e.predicted_cost,
                confidence=e.confidence,
                agent_scores=AgentScore(performance=p, efficiency=ef, stability=st),
                final_score=final,
                rationale=rationale,
            ))

        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked
    except Exception as exc:
        log.warning("coordinator_agent LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def rank(
    state: StateObject,
    estimates: list[UncertaintyEstimate],
    perf_scores: dict[str, float],
    eff_scores: dict[str, float],
    stab_scores: dict[str, float],
    log_paths: list[str | Path] | None = None,
) -> list[RankedAction]:
    """Return candidates sorted by final_score descending."""
    if os.getenv("GROQ_API_KEY"):
        result = _llm_rank(state, estimates, perf_scores, eff_scores, stab_scores)
        if result is not None:
            return result
    return _rule_rank(state, estimates, perf_scores, eff_scores, stab_scores, log_paths=log_paths)
