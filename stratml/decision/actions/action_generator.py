"""
action_generator.py
-------------------
Decision/Actions — Candidate Action Generator.

LLM path: LangChain chain reads full StateObject and proposes list[CandidateAction]
          via structured output. Can propose context-sensitive, composed actions.
Fallback: rule-based if-else tree (used when LLM fails or GROQ_API_KEY absent).

Bootstrap (iteration 0) always uses the rule-based path — no LLM needed there.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel

from stratml.core.schemas import CandidateAction, StateObject
from stratml.decision.llm_control import is_llm_enabled
from stratml.execution.pipelines.ml_pipeline import (
    PAPER_CLASSIFICATION_MODELS,
    PAPER_REGRESSION_MODELS,
)

log = logging.getLogger(__name__)

_DEFAULT_MODELS = PAPER_CLASSIFICATION_MODELS
_DEFAULT_REGRESSION_MODELS = PAPER_REGRESSION_MODELS

_BOOTSTRAP_MODELS = _DEFAULT_MODELS


def _get_default_models(state: StateObject) -> list[str]:
    is_reg = state.objective.primary_metric in ("r2", "mse", "rmse") or (
        state.dataset and getattr(state.dataset, "problem_type", None) == "regression"
    )
    return _DEFAULT_REGRESSION_MODELS if is_reg else _DEFAULT_MODELS


PAPER_CLASSICAL_ACTIONS: set[str] = {
    "switch_model",
    "increase_model_capacity",
    "decrease_model_capacity",
    "modify_regularization",
    "add_preprocessing",
    "terminate",
}

_VALID_ACTION_TYPES = {
    "switch_model",
    "increase_model_capacity",
    "decrease_model_capacity",
    "modify_regularization",
    "change_optimizer",
    "unfreeze_backbone",
    "switch_architecture",
    "early_stop",
    "add_preprocessing",
    "terminate",
}

_DL_VISION_MODELS  = ["CNN2D", "ResNet18", "EfficientNetB0", "MobileNetV3"]
_DL_TEXT_MODELS    = ["TextCNN", "BiLSTM", "DistilBERT", "TinyBERT"]
_DL_TABULAR_MODELS = ["MLP", "CNN1D", "RNN", "ResidualMLP", "TabTransformer"]

_DL_TOO_SLOW_FALLBACK = {
    "vision": "MobileNetV3",
    "text":   "TinyBERT",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(state: StateObject) -> list[CandidateAction]:
    """Return candidate actions for the current state."""
    is_dl = getattr(state.model, "model_type", "ml") == "dl"
    if state.resources.budget_exhausted:
        result = [CandidateAction(action_type="terminate", parameters={})]
    elif state.meta.iteration == 0:
        result = _bootstrap_candidates(state)
    elif is_llm_enabled():
        result = _llm_candidates(state)
        if result is None:
            result = _rule_candidates(state)
    else:
        result = _rule_candidates(state)

    if not is_dl:
        result = [c for c in result if c.action_type in PAPER_CLASSICAL_ACTIONS]
        if not any(c.action_type == "terminate" for c in result):
            result.append(CandidateAction(action_type="terminate", parameters={}))
    return result


# ---------------------------------------------------------------------------
# Bootstrap (iteration 0)
# ---------------------------------------------------------------------------

def _bootstrap_candidates(state: StateObject) -> list[CandidateAction]:
    allowed = state.constraints.allowed_models or _get_default_models(state)
    return [
        CandidateAction(action_type="switch_model", parameters={"model_name": m})
        for m in allowed[:2]
    ]


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_CLASSICAL_SYSTEM_PROMPT = (
    "You are an ML experimentation strategist. Given the current experiment state, "
    "propose a list of candidate next actions. You may suggest actions the rules don't "
    "cover, compose specific parameter values (e.g., a concrete alpha for regularization "
    "derived from the gap magnitude), or recommend a specific model based on dataset "
    "characteristics. Always include 'terminate' as one candidate — it is a valid choice "
    "at any point if the situation warrants it. "
    "Valid action_types: switch_model, increase_model_capacity, decrease_model_capacity, "
    "modify_regularization, add_preprocessing, terminate."
)

_DL_SYSTEM_PROMPT = (
    "You are an ML experimentation strategist. Given the current experiment state, "
    "propose a list of candidate next actions. You may suggest actions the rules don't "
    "cover, compose specific parameter values, or recommend a specific architecture. "
    "Always include 'terminate' as one candidate. "
    "Valid action_types: switch_model, increase_model_capacity, decrease_model_capacity, "
    "modify_regularization, change_optimizer, unfreeze_backbone, switch_architecture, add_preprocessing, terminate."
)

_SYSTEM_PROMPT = _CLASSICAL_SYSTEM_PROMPT


class _CandidateItem(BaseModel):
    action_type: str
    parameters: dict


class _CandidateList(BaseModel):
    candidates: list[_CandidateItem]


def _llm_candidates(state: StateObject) -> Optional[list[CandidateAction]]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        sig = state.signals
        traj = state.trajectory
        allowed = state.constraints.allowed_models or _get_default_models(state)
        tried = state.search.models_tried
        untried = [m for m in allowed if m not in tried]

        human = (
            f"Iteration: {state.meta.iteration}\n"
            f"Model: {state.model.model_name} ({state.model.model_type})\n"
            f"Dataset: {state.dataset.num_samples} samples, {state.dataset.num_features} features, "
            f"imbalance_ratio={state.dataset.imbalance_ratio}\n"
            f"Fitting: underfitting={sig.underfitting}, overfitting={sig.overfitting}, "
            f"well_fitted={sig.well_fitted}, plateau={sig.plateau_detected}\n"
            f"Trajectory: trend={traj.trend}, slope={traj.slope:.4f}, "
            f"steps_since_improvement={traj.steps_since_improvement}, best_score={traj.best_score:.4f}\n"
            f"Resources: remaining_budget={state.resources.remaining_budget}, "
            f"budget_exhausted={state.resources.budget_exhausted}\n"
            f"Allowed models: {allowed}\n"
            f"Untried models: {untried}\n"
            f"Models tried so far: {tried}\n"
            f"Train/val gap: {state.generalization.gap:.4f}\n\n"
            "Propose 2-4 candidate actions as a JSON list. Each must have action_type and parameters."
        )

        is_dl = getattr(state.model, "model_type", "ml") == "dl"
        prompt = _DL_SYSTEM_PROMPT if is_dl else _CLASSICAL_SYSTEM_PROMPT

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2).with_structured_output(_CandidateList)
        output: _CandidateList = llm.invoke([SystemMessage(prompt), HumanMessage(human)])

        candidates = [
            CandidateAction(action_type=item.action_type, parameters=item.parameters)
            for item in output.candidates
            if (item.action_type in _VALID_ACTION_TYPES if is_dl else item.action_type in PAPER_CLASSICAL_ACTIONS)
        ]

        if not candidates:
            return None

        if not any(c.action_type == "terminate" for c in candidates):
            candidates.append(CandidateAction(action_type="terminate", parameters={}))

        # Deduplicate
        seen: set[tuple] = set()
        unique: list[CandidateAction] = []
        for c in candidates:
            key = (c.action_type, str(sorted(c.parameters.items())))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique
    except Exception as exc:
        log.warning("action_generator LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback (iteration 1+)
# ---------------------------------------------------------------------------

def _rule_candidates(state: StateObject) -> list[CandidateAction]:
    sig = state.signals
    candidates: list[CandidateAction] = []

    model_type = getattr(state.model, "model_type", "ml")
    modality   = getattr(state.model, "modality", "tabular") if hasattr(state.model, "modality") else "tabular"

    if model_type == "dl":
        return _rule_candidates_dl(state, modality)

    allowed = state.constraints.allowed_models or _get_default_models(state)
    tried = set(state.search.models_tried)
    untried = [m for m in allowed if m not in tried]

    if state.resources.budget_exhausted:
        return [CandidateAction(action_type="terminate", parameters={})]

    if not untried and state.meta.iteration > 1 and sig.converged != "none" and sig.well_fitted != "none":
        return [CandidateAction(action_type="terminate", parameters={})]

    if sig.underfitting != "none":
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))
        candidates.append(CandidateAction(action_type="increase_model_capacity", parameters={"scale": 1.5}))

    if sig.overfitting != "none":
        # Always offer an untried model — regularization alone rarely fixes structural overfitting
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))
        candidates.append(CandidateAction(action_type="modify_regularization", parameters={"direction": "increase"}))
        candidates.append(CandidateAction(action_type="decrease_model_capacity", parameters={"scale": 0.75}))
        if state.dataset.imbalance_ratio and state.dataset.imbalance_ratio > 2.0:
            candidates.append(CandidateAction(action_type="add_preprocessing", parameters={"strategy": "oversample"}))

    if sig.stagnating != "none" or sig.plateau_detected != "none":
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))

    if sig.diverging != "none":
        candidates.append(CandidateAction(action_type="modify_regularization", parameters={"direction": "increase"}))
        candidates.append(CandidateAction(action_type="decrease_model_capacity", parameters={"scale": 0.75}))
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))

    if sig.diminishing_returns != "none":
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))

    # No signal fired — pure exploration
    if not candidates and untried:
        candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))

    # terminate is always a valid candidate — coordinator decides whether to pick it
    if not any(c.action_type == "terminate" for c in candidates):
        candidates.append(CandidateAction(action_type="terminate", parameters={}))

    # Deduplicate
    seen: set[tuple] = set()
    unique: list[CandidateAction] = []
    for c in candidates:
        key = (c.action_type, str(sorted(c.parameters.items())))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# DL-aware rule candidates
# ---------------------------------------------------------------------------

def _rule_candidates_dl(state: StateObject, modality: str) -> list[CandidateAction]:
    """Rule candidates when model_type == 'dl'. Respects modality pools."""
    sig = state.signals
    candidates: list[CandidateAction] = []

    if modality == "vision":
        pool = _DL_VISION_MODELS
    elif modality == "text":
        pool = _DL_TEXT_MODELS
    else:
        pool = _DL_TABULAR_MODELS

    tried   = set(state.search.models_tried)
    untried = [m for m in pool if m not in tried]

    if state.resources.budget_exhausted:
        return [CandidateAction(action_type="terminate", parameters={})]

    if not untried and state.meta.iteration > 1 and sig.converged != "none" and sig.well_fitted != "none":
        return [CandidateAction(action_type="terminate", parameters={})]

    # too_slow: drop to the lightweight pretrained model for this modality
    if sig.too_slow != "none":
        fallback = _DL_TOO_SLOW_FALLBACK.get(modality)
        if fallback and fallback not in tried:
            candidates.append(CandidateAction(
                action_type="switch_model", parameters={"model_name": fallback}
            ))

    if sig.underfitting != "none":
        if untried:
            candidates.append(CandidateAction(
                action_type="switch_model", parameters={"model_name": untried[0]}
            ))
        candidates.append(CandidateAction(
            action_type="increase_model_capacity", parameters={"scale": 1.5}
        ))

    if sig.overfitting != "none":
        candidates.append(CandidateAction(
            action_type="modify_regularization", parameters={"direction": "increase"}
        ))
        candidates.append(CandidateAction(
            action_type="decrease_model_capacity", parameters={"scale": 0.75}
        ))

    if sig.stagnating != "none" or sig.plateau_detected != "none":
        # Escalate to pretrained tier
        if untried:
            candidates.append(CandidateAction(
                action_type="switch_model", parameters={"model_name": untried[0]}
            ))

    if sig.diverging != "none":
        candidates.append(CandidateAction(
            action_type="change_optimizer", parameters={"learning_rate_scale": 0.1}
        ))

    # Progressive unfreezing when still underfitting after pretrained model
    current_arch = getattr(state.model, "model_name", "")
    _pretrained = {"ResNet18", "EfficientNetB0", "MobileNetV3", "DistilBERT", "TinyBERT"}
    if current_arch in _pretrained and sig.underfitting != "none":
        candidates.append(CandidateAction(
            action_type="unfreeze_backbone", parameters={"n_layers": 1}
        ))

    if not candidates and untried:
        candidates.append(CandidateAction(
            action_type="switch_model", parameters={"model_name": untried[0]}
        ))

    if not any(c.action_type == "terminate" for c in candidates):
        candidates.append(CandidateAction(action_type="terminate", parameters={}))

    # Deduplicate
    seen: set[tuple] = set()
    unique: list[CandidateAction] = []
    for c in candidates:
        key = (c.action_type, str(sorted(c.parameters.items())))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique
