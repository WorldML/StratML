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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from stratml.core.schemas import AgentScore, StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate
from stratml.decision.llm_control import is_llm_enabled

log = logging.getLogger(__name__)

_W_PERF_DEFAULT = 0.50
_W_EFF_DEFAULT  = 0.25
_W_STAB_DEFAULT = 0.25
_EMA_ALPHA = 0.2

_CURRENT_WEIGHTS: dict[str, float] = {
    "performance_weight": _W_PERF_DEFAULT,
    "efficiency_weight": _W_EFF_DEFAULT,
    "stability_weight": _W_STAB_DEFAULT,
}

_WEIGHT_UPDATE_HISTORY: list[dict] = []


def get_current_weights() -> dict[str, float]:
    """Return the coordinator weights used for the latest ranking."""
    return dict(_CURRENT_WEIGHTS)


def get_persisted_weight_updates(path: str | Path) -> list[dict]:
    """Load persisted coordinator weight updates from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_weight_update_history(log_path: str | Path | None = None) -> list[dict]:
    """Return the attributable audit trail of coordinator weight updates."""
    if log_path is not None:
        return get_persisted_weight_updates(log_path)
    return [dict(u) for u in _WEIGHT_UPDATE_HISTORY]


def reset_weight_update_history() -> None:
    """Reset coordinator weight update history to defaults."""
    global _WEIGHT_UPDATE_HISTORY, _CURRENT_WEIGHTS
    _WEIGHT_UPDATE_HISTORY = []
    _CURRENT_WEIGHTS = {
        "performance_weight": _W_PERF_DEFAULT,
        "efficiency_weight": _W_EFF_DEFAULT,
        "stability_weight": _W_STAB_DEFAULT,
    }


def _set_current_weights(w_p: float, w_e: float, w_s: float) -> None:
    global _CURRENT_WEIGHTS
    _CURRENT_WEIGHTS = {
        "performance_weight": w_p,
        "efficiency_weight": w_e,
        "stability_weight": w_s,
    }


def _get_update_log_path(
    log_paths: list[str | Path] | None = None,
    update_log_path: str | Path | None = None,
    run_id: str | None = None,
) -> Path | None:
    if update_log_path is not None:
        return Path(update_log_path)
    if log_paths:
        first = Path(log_paths[0])
        return first.parent / "coordinator_weight_updates.jsonl"
    if run_id:
        return Path("outputs") / run_id / "decision_logs" / "coordinator_weight_updates.jsonl"
    return None


def _load_agent_weights(
    log_paths: list[str | Path] | None = None,
    update_log_path: str | Path | None = None,
    run_id: str | None = None,
) -> tuple[float, float, float]:
    """Compute per-agent EMA weights deterministically with full attributable audit trail and artifact persistence."""
    import glob
    from pathlib import Path
    if log_paths is not None:
        logs = sorted([str(p) for p in log_paths])
    else:
        logs = sorted(glob.glob("outputs/*/decision_logs/evaluation_log.jsonl"))

    if not logs:
        weights = (_W_PERF_DEFAULT, _W_EFF_DEFAULT, _W_STAB_DEFAULT)
        _set_current_weights(*weights)
        return weights

    try:
        import json
        records = []
        for log_path in logs:
            with open(log_path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["_source_log"] = str(log_path)
                        rec["_line_index"] = idx
                        records.append(rec)

        if len(records) < 5:
            weights = (_W_PERF_DEFAULT, _W_EFF_DEFAULT, _W_STAB_DEFAULT)
            _set_current_weights(*weights)
            return weights

        # Deterministic EMA weight updates with bounded values and attributable history
        w_p = w_e = w_s = 0.5
        _WEIGHT_UPDATE_HISTORY.clear()

        persist_path = _get_update_log_path(log_paths=logs, update_log_path=update_log_path, run_id=run_id)

        for idx, r in enumerate(records):
            eval_id = r.get("evaluation_id") or r.get("experiment_id") or f"eval_{idx}"
            iteration = r.get("iteration")
            rec_run_id = r.get("run_id") or run_id
            if not rec_run_id and "_source_log" in r:
                src_parts = Path(r["_source_log"]).parts
                if len(src_parts) >= 3 and src_parts[-3] != "outputs":
                    rec_run_id = src_parts[-3]
            if not rec_run_id:
                rec_run_id = str(eval_id)

            prev_total = w_p + w_e + w_s
            prev_norm = {
                "performance_weight": round(w_p / prev_total, 4),
                "efficiency_weight": round(w_e / prev_total, 4),
                "stability_weight": round(w_s / prev_total, 4),
            }

            cf = float(r.get("counterfactual_impact", 0.0) or 0.0)
            validity = float(r.get("decision_validity", 0.5) or 0.5)
            risk = float(r.get("quality_risk", 0.5) or 0.5)

            perf_right = 1.0 if (validity >= 0.5 and cf >= 0.0) else 0.0
            eff_right  = 1.0 if cf >= -0.02 else 0.0
            stab_right = 1.0 if risk < 0.5 else 0.0

            evidence = {
                "counterfactual_impact": cf,
                "decision_validity": validity,
                "quality_risk": risk,
                "perf_agent_correct": perf_right,
                "eff_agent_correct": eff_right,
                "stab_agent_correct": stab_right,
            }

            # Apply bounded EMA update (minimum 1e-4 so weights never collapse to zero)
            w_p = max(1e-4, (1.0 - _EMA_ALPHA) * w_p + _EMA_ALPHA * perf_right)
            w_e = max(1e-4, (1.0 - _EMA_ALPHA) * w_e + _EMA_ALPHA * eff_right)
            w_s = max(1e-4, (1.0 - _EMA_ALPHA) * w_s + _EMA_ALPHA * stab_right)

            new_total = w_p + w_e + w_s
            new_norm = {
                "performance_weight": round(w_p / new_total, 4),
                "efficiency_weight": round(w_e / new_total, 4),
                "stability_weight": round(w_s / new_total, 4),
            }

            entry = {
                "update_index": idx,
                "evaluation_id": str(eval_id),
                "run_id": str(rec_run_id),
                "iteration": iteration,
                "previous_weights": {
                    "performance": prev_norm["performance_weight"],
                    "efficiency": prev_norm["efficiency_weight"],
                    "stability": prev_norm["stability_weight"],
                    "performance_weight": prev_norm["performance_weight"],
                    "efficiency_weight": prev_norm["efficiency_weight"],
                    "stability_weight": prev_norm["stability_weight"],
                },
                "evidence": evidence,
                "unnormalized_weights": {
                    "performance": round(w_p, 6),
                    "efficiency": round(w_e, 6),
                    "stability": round(w_s, 6),
                },
                "new_weights": {
                    "performance": new_norm["performance_weight"],
                    "efficiency": new_norm["efficiency_weight"],
                    "stability": new_norm["stability_weight"],
                    "performance_weight": new_norm["performance_weight"],
                    "efficiency_weight": new_norm["efficiency_weight"],
                    "stability_weight": new_norm["stability_weight"],
                },
            }
            _WEIGHT_UPDATE_HISTORY.append(entry)

        # Persist attributable updates to append-only JSONL artifact
        if persist_path is not None:
            try:
                persist_path.parent.mkdir(parents=True, exist_ok=True)
                existing_keys = set()
                if persist_path.exists():
                    with open(persist_path, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            line = line.strip()
                            if line:
                                item = json.loads(line)
                                existing_keys.add((
                                    str(item.get("evaluation_id")),
                                    item.get("iteration"),
                                    str(item.get("run_id")),
                                ))
                with open(persist_path, "a", encoding="utf-8") as f_out:
                    for entry in _WEIGHT_UPDATE_HISTORY:
                        key = (str(entry.get("evaluation_id")), entry.get("iteration"), str(entry.get("run_id")))
                        if key not in existing_keys:
                            record_to_write = {
                                "evaluation_id": entry["evaluation_id"],
                                "run_id": entry["run_id"],
                                "iteration": entry["iteration"],
                                "previous_weights": {
                                    "performance": entry["previous_weights"]["performance"],
                                    "efficiency": entry["previous_weights"]["efficiency"],
                                    "stability": entry["previous_weights"]["stability"],
                                },
                                "evidence": entry["evidence"],
                                "new_weights": {
                                    "performance": entry["new_weights"]["performance"],
                                    "efficiency": entry["new_weights"]["efficiency"],
                                    "stability": entry["new_weights"]["stability"],
                                },
                            }
                            f_out.write(json.dumps(record_to_write) + "\n")
                            existing_keys.add(key)
            except Exception as e:
                log.warning("Failed to persist coordinator weight updates to %s: %s", persist_path, e)

        total = w_p + w_e + w_s
        weights = (round(w_p / total, 4), round(w_e / total, 4), round(w_s / total, 4))
        _set_current_weights(*weights)
        return weights
    except Exception as exc:
        log.warning("Coordinator weight loading failed: %s", exc)
        weights = (_W_PERF_DEFAULT, _W_EFF_DEFAULT, _W_STAB_DEFAULT)
        _set_current_weights(*weights)
        return weights


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
    source: Optional[str] = field(default=None)


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
    source: str = "rule",
    update_log_path: str | Path | None = None,
    run_id: str | None = None,
) -> list[RankedAction]:
    if log_paths is not None or update_log_path is not None or run_id is not None:
        w_p, w_e, w_s = _load_agent_weights(log_paths=log_paths, update_log_path=update_log_path, run_id=run_id)
    else:
        w_p, w_e, w_s = _load_agent_weights()
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
            source=source,
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
            final = round(max(0.0, min(llm_item.final_score, 1.0)), 4) if llm_item else round(_W_PERF_DEFAULT * p + _W_EFF_DEFAULT * ef + _W_STAB_DEFAULT * st, 4)
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
                source="llm",
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
    source: Optional[str] = None,
    has_learned_model: bool = False,
    update_log_path: str | Path | None = None,
    run_id: str | None = None,
) -> list[RankedAction]:
    """Return candidates sorted by final_score descending."""
    if is_llm_enabled():
        result = _llm_rank(state, estimates, perf_scores, eff_scores, stab_scores)
        if result is not None:
            return result
        # Explicit fallback source when LLM was requested/enabled but unavailable
        return _rule_rank(
            state,
            estimates,
            perf_scores,
            eff_scores,
            stab_scores,
            log_paths=log_paths,
            source="fallback",
            update_log_path=update_log_path,
            run_id=run_id,
        )

    effective_source = source
    if effective_source is None:
        effective_source = "hybrid" if has_learned_model else "rule"

    return _rule_rank(
        state,
        estimates,
        perf_scores,
        eff_scores,
        stab_scores,
        log_paths=log_paths,
        source=effective_source,
        update_log_path=update_log_path,
        run_id=run_id,
    )
