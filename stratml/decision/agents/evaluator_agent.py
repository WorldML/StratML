"""
evaluator_agent.py
------------------
Decision Council — Evaluator Agent (Post-Hoc Decision Auditor).

Runs inside receive_result() AFTER the new state is built.
Audits the PREVIOUS decision against the outcome that just arrived.

Four dimensions:
    decision_validity      — was the action type appropriate for the signals present?
    reasoning_consistency  — did the trigger match the actual signals?
    quality_risk           — generalization gap / instability concerns?
    counterfactual_impact  — actual gain vs. expected gain

Output: EvaluationRecord appended to outputs/<run_id>/decision_logs/evaluation_log.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from stratml.core.schemas import ActionDecision, ExperimentResult, StateObject
from stratml.decision.llm_control import is_llm_enabled

log = logging.getLogger(__name__)

_EVAL_LOG = Path("runs/decision_logs/evaluation_log.jsonl")  # overridden by engine

# Expected action types per trigger (rule-based validity check)
_TRIGGER_EXPECTED: dict[str, set[str]] = {
    "underfitting":        {"switch_model", "increase_model_capacity"},
    "overfitting":         {"switch_model", "modify_regularization", "decrease_model_capacity", "add_preprocessing"},
    "stagnation":          {"switch_model", "change_optimizer"},
    "divergence":          {"change_optimizer", "decrease_model_capacity"},
    "diminishing_returns": {"switch_model"},
    "convergence":         {"terminate"},
    "bootstrap":           {"switch_model"},
    "exploration":         {"switch_model", "increase_model_capacity", "modify_regularization"},
}


@dataclass
class EvaluationRecord:
    experiment_id: str
    iteration: int
    action_type: str
    trigger: str
    decision_validity: float        # 0.0–1.0
    reasoning_consistency: float    # 0.0–1.0
    quality_risk: float             # 0.0–1.0 (higher = more risk)
    counterfactual_impact: float    # actual_gain - expected_gain
    fault_detected: bool
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Rule-based audit
# ---------------------------------------------------------------------------

def _rule_audit(decision: ActionDecision, result: ExperimentResult, state: StateObject) -> EvaluationRecord:
    trigger = decision.reason.trigger

    # Decision validity: is the action in the expected set for this trigger?
    expected = _TRIGGER_EXPECTED.get(trigger, set())
    validity = 1.0 if (not expected or decision.action_type in expected) else 0.0

    # Reasoning consistency: is the trigger's signal actually present in the new state?
    sig = state.signals
    signal_present = {
        "underfitting":        sig.underfitting != "none",
        "overfitting":         sig.overfitting != "none",
        "stagnation":          sig.stagnating != "none" or sig.plateau_detected != "none",
        "divergence":          sig.diverging != "none",
        "diminishing_returns": sig.diminishing_returns != "none",
        "convergence":         sig.converged != "none" and sig.well_fitted != "none",
        "bootstrap":           True,
        "exploration":         True,
    }
    consistency = 1.0 if signal_present.get(trigger, True) else 0.3

    # Quality risk: gap magnitude + instability
    gap = abs(state.generalization.gap)
    risk = min(1.0, gap / 0.2 * 0.6 + (0.4 if sig.unstable_training != "none" else 0.0))

    # Counterfactual impact: actual gain vs expected
    cf_impact = _compute_cf_impact(decision, result, state)

    fault = validity < 0.5 or consistency < 0.5
    notes = []
    if validity < 0.5:
        notes.append(f"action '{decision.action_type}' unexpected for trigger '{trigger}'")
    if consistency < 0.5:
        notes.append(f"trigger '{trigger}' signal not present in state")

    return EvaluationRecord(
        experiment_id=decision.experiment_id,
        iteration=decision.iteration,
        action_type=decision.action_type,
        trigger=trigger,
        decision_validity=round(validity, 4),
        reasoning_consistency=round(consistency, 4),
        quality_risk=round(risk, 4),
        counterfactual_impact=cf_impact,
        fault_detected=fault,
        notes="; ".join(notes),
    )


def _compute_cf_impact(decision: ActionDecision, result: ExperimentResult, state: StateObject) -> float:
    _metric_name = decision.reason.evidence.get("primary_metric") or "accuracy"
    if not isinstance(_metric_name, str):
        _metric_name = "accuracy"
    primary = getattr(result.metrics, _metric_name, None)
    if primary is None:
        primary = 0.0
    previous_best = decision.reason.evidence.get("best_score")
    if previous_best is None:
        previous_best = getattr(state.trajectory, "best_score", 0.0) or 0.0
    actual_gain = primary - previous_best
    expected_gain = decision.expected_gain or 0.0
    return round(actual_gain - expected_gain, 4)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a post-hoc decision auditor for an ML experimentation system. "
    "Given the decision that was made and the outcome that resulted, score four dimensions "
    "on a 0.0-1.0 scale: decision_validity (was the action appropriate?), "
    "reasoning_consistency (did the trigger match the signals?), "
    "quality_risk (0=no risk, 1=high risk from gap/instability), "
    "counterfactual_impact (actual_gain minus expected_gain, can be negative). "
    "Also set fault_detected=true if validity or consistency < 0.5. "
    "Return JSON with keys: decision_validity, reasoning_consistency, quality_risk, "
    "counterfactual_impact, fault_detected, notes."
)


def _llm_audit(decision: ActionDecision, result: ExperimentResult, state: StateObject):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import BaseModel

        class _AuditOutput(BaseModel):
            decision_validity: float
            reasoning_consistency: float
            quality_risk: float
            counterfactual_impact: float
            fault_detected: bool
            notes: str = ""

        sig = state.signals
        human = (
            f"Decision: action={decision.action_type}, trigger={decision.reason.trigger}, "
            f"expected_gain={decision.expected_gain}, confidence={decision.confidence}\n"
            f"Outcome: primary_metric={getattr(result.metrics, 'accuracy', None)}, "
            f"train_loss={result.metrics.train_loss}, val_loss={result.metrics.validation_loss}\n"
            f"State signals: underfitting={sig.underfitting}, overfitting={sig.overfitting}, "
            f"well_fitted={sig.well_fitted}, stagnating={sig.stagnating}, "
            f"converged={sig.converged}, diverging={sig.diverging}\n"
            f"Generalization gap: {state.generalization.gap:.4f}\n"
            f"Previous signals: {decision.reason.evidence.get('previous_signals', 'N/A')}\n"
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(_AuditOutput)
        out: _AuditOutput = llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human)])

        cf_impact = _compute_cf_impact(decision, result, state)

        return EvaluationRecord(
            experiment_id=decision.experiment_id,
            iteration=decision.iteration,
            action_type=decision.action_type,
            trigger=decision.reason.trigger,
            decision_validity=round(max(0.0, min(out.decision_validity, 1.0)), 4),
            reasoning_consistency=round(max(0.0, min(out.reasoning_consistency, 1.0)), 4),
            quality_risk=round(max(0.0, min(out.quality_risk, 1.0)), 4),
            counterfactual_impact=cf_impact,
            fault_detected=out.fault_detected,
            notes=out.notes,
        )
    except Exception as exc:
        log.warning("evaluator_agent LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def audit(decision: ActionDecision, result: ExperimentResult, state: StateObject) -> EvaluationRecord:
    """Audit the previous decision against the new result. Appends to evaluation_log.jsonl."""
    rec = None
    if is_llm_enabled():
        rec = _llm_audit(decision, result, state)
    if rec is None:
        rec = _rule_audit(decision, result, state)

    _EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_EVAL_LOG, "a") as f:
        f.write(json.dumps({
            "experiment_id": rec.experiment_id,
            "iteration": rec.iteration,
            "action_type": rec.action_type,
            "trigger": rec.trigger,
            "decision_validity": rec.decision_validity,
            "reasoning_consistency": rec.reasoning_consistency,
            "quality_risk": rec.quality_risk,
            "counterfactual_impact": rec.counterfactual_impact,
            "fault_detected": rec.fault_detected,
            "notes": rec.notes,
        }) + "\n")

    if rec.fault_detected:
        log.warning(
            "evaluator_agent: fault detected at iteration %d — %s",
            rec.iteration, rec.notes or "see evaluation_log.jsonl"
        )

    return rec
