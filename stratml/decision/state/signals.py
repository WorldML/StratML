"""
signals.py
----------
Phase 2 (Dev B) — Signal Extraction.

Uses a LangChain ReAct agent backed by GPT-4o-mini to reason over
StateObject metrics and call signal-flagging tools.  Each tool covers
one signal group and returns a (strength, confidence) pair.

Falls back to rule-based logic if the agent errors or is unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from stratml.decision.llm_control import is_llm_enabled

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        fn.func = fn
        return fn

try:
    from langgraph.prebuilt import create_react_agent
    from langchain_groq import ChatGroq
except ImportError:
    create_react_agent = None
    ChatGroq = None

from stratml.core.schemas import StateObject, StateSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strength(weak: bool, strong: bool) -> str:
    if strong:
        return "strong"
    if weak:
        return "weak"
    return "none"


def _clamp(v: float) -> float:
    return round(max(0.0, min(float(v), 1.0)), 4)


# ---------------------------------------------------------------------------
# Signal tools — called by the agent
# ---------------------------------------------------------------------------

@tool
def assess_fitting(primary: float, gap: float) -> str:
    """
    Assess underfitting, overfitting, and well_fitted signals.

    Args:
        primary: Current primary metric score (e.g. accuracy), 0–1.
        gap:     val_loss - train_loss (generalisation gap).

    Returns JSON with keys: underfitting, underfitting_confidence,
    overfitting, overfitting_confidence, well_fitted, well_fitted_confidence.
    Each signal is "none" | "weak" | "strong".
    """
    uf_strong = primary < 0.60
    uf_weak   = 0.60 <= primary < 0.70
    of_strong = gap > 0.10
    of_weak   = 0.05 < gap <= 0.10
    not_uf    = not (uf_weak or uf_strong)
    not_of    = not (of_weak or of_strong)
    wf_strong = not_uf and not_of and primary >= 0.75
    wf_weak   = not_uf and not_of and 0.70 <= primary < 0.75

    return json.dumps({
        "underfitting":            _strength(uf_weak, uf_strong),
        "underfitting_confidence": _clamp((0.70 - primary) / 0.70),
        "overfitting":             _strength(of_weak, of_strong),
        "overfitting_confidence":  _clamp(gap / 0.20),
        "well_fitted":             _strength(wf_weak, wf_strong),
        "well_fitted_confidence":  _clamp(primary / 0.90) if (wf_weak or wf_strong) else 0.0,
    })


@tool
def assess_convergence(slope: float, primary: float, steps_since: int) -> str:
    """
    Assess converged, stagnating, and diverging signals.

    Args:
        slope:       Trend slope of the primary metric over recent history.
        primary:     Current primary metric score.
        steps_since: Iterations since last improvement.

    Returns JSON with keys: converged, converged_confidence,
    stagnating, stagnating_confidence, diverging, diverging_confidence.
    """
    cv_strong = abs(slope) < 0.001 and primary >= 0.75
    cv_weak   = abs(slope) < 0.005 and primary >= 0.70
    sg_strong = steps_since >= 4 and not (cv_weak or cv_strong)
    sg_weak   = steps_since >= 2 and not (cv_weak or cv_strong)
    dv_strong = slope < -0.02
    dv_weak   = -0.02 <= slope < -0.01

    return json.dumps({
        "converged":            _strength(cv_weak, cv_strong),
        "converged_confidence": _clamp(1.0 - abs(slope) / 0.005) if (cv_weak or cv_strong) else 0.0,
        "stagnating":            _strength(sg_weak, sg_strong),
        "stagnating_confidence": _clamp(steps_since / 5.0) if (sg_weak or sg_strong) else 0.0,
        "diverging":             _strength(dv_weak, dv_strong),
        "diverging_confidence":  _clamp(abs(slope) / 0.05) if (dv_weak or dv_strong) else 0.0,
    })


@tool
def assess_stability(train_loss: float, val_loss: float, volatility: float) -> str:
    """
    Assess unstable_training and high_variance signals.

    Args:
        train_loss:  Training loss from the last run.
        val_loss:    Validation loss from the last run.
        volatility:  Standard deviation of recent primary metric scores.

    Returns JSON with keys: unstable_training, unstable_training_confidence,
    high_variance, high_variance_confidence.
    """
    ratio     = val_loss / train_loss if train_loss > 0 else 0.0
    ut_strong = ratio > 2.0
    ut_weak   = 1.5 < ratio <= 2.0
    hv_strong = volatility > 0.05
    hv_weak   = 0.03 < volatility <= 0.05

    return json.dumps({
        "unstable_training":            _strength(ut_weak, ut_strong),
        "unstable_training_confidence": _clamp((ratio - 1.0) / 2.0) if (ut_weak or ut_strong) else 0.0,
        "high_variance":                _strength(hv_weak, hv_strong),
        "high_variance_confidence":     _clamp(volatility / 0.10) if (hv_weak or hv_strong) else 0.0,
    })


@tool
def assess_efficiency(runtime: float) -> str:
    """
    Assess the too_slow signal.

    Args:
        runtime: Wall-clock runtime of the last experiment in seconds.

    Returns JSON with keys: too_slow, too_slow_confidence.
    """
    ts_strong = runtime > 300.0
    ts_weak   = 200.0 < runtime <= 300.0

    return json.dumps({
        "too_slow":            _strength(ts_weak, ts_strong),
        "too_slow_confidence": _clamp(runtime / 600.0) if (ts_weak or ts_strong) else 0.0,
    })


@tool
def assess_optimization(steps_since: int, improvement_rate: float) -> str:
    """
    Assess plateau_detected and diminishing_returns signals.

    Args:
        steps_since:      Iterations since last improvement.
        improvement_rate: Delta of primary metric from last iteration.

    Returns JSON with keys: plateau_detected, plateau_detected_confidence,
    diminishing_returns, diminishing_returns_confidence.
    """
    pl_strong = steps_since >= 4
    pl_weak   = steps_since == 3
    dr_strong = 0.0 < improvement_rate < 0.005
    dr_weak   = 0.005 <= improvement_rate < 0.01

    return json.dumps({
        "plateau_detected":             _strength(pl_weak, pl_strong),
        "plateau_detected_confidence":  _clamp(steps_since / 5.0) if (pl_weak or pl_strong) else 0.0,
        "diminishing_returns":          _strength(dr_weak, dr_strong),
        "diminishing_returns_confidence": _clamp(1.0 - improvement_rate / 0.01) if (dr_weak or dr_strong) else 0.0,
    })


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

_TOOLS = [assess_fitting, assess_convergence, assess_stability, assess_efficiency, assess_optimization]

_AGENT = None

def _get_agent() -> object:
    global _AGENT
    if _AGENT is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        _AGENT = create_react_agent(llm, _TOOLS)
    return _AGENT


# ---------------------------------------------------------------------------
# Rule-based fallback (no LLM)
# ---------------------------------------------------------------------------

def _rule_based(state: StateObject) -> StateSignals:
    g = state.generalization
    t = state.trajectory
    r = state.resources
    primary = state.metrics.primary

    fit  = json.loads(assess_fitting.func(primary, g.gap))
    conv = json.loads(assess_convergence.func(t.slope, primary, t.steps_since_improvement))
    stab = json.loads(assess_stability.func(g.train_loss, g.validation_loss, t.volatility))
    eff  = json.loads(assess_efficiency.func(r.runtime))
    opt  = json.loads(assess_optimization.func(t.steps_since_improvement, t.improvement_rate))

    return StateSignals(**{**fit, **conv, **stab, **eff, **opt})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_signals(state: StateObject) -> StateSignals:
    """
    Use a ReAct agent to reason over state metrics and flag signals.
    Falls back to direct rule evaluation if the agent is unavailable.
    """
    if not is_llm_enabled():
        return _rule_based(state)

    g = state.generalization
    t = state.trajectory
    r = state.resources

    metrics_summary = (
        f"primary={state.metrics.primary}, gap={g.gap}, "
        f"train_loss={g.train_loss}, val_loss={g.validation_loss}, "
        f"slope={t.slope}, steps_since_improvement={t.steps_since_improvement}, "
        f"improvement_rate={t.improvement_rate}, volatility={t.volatility}, "
        f"runtime={r.runtime}"
    )

    try:
        agent   = _get_agent()
        result  = agent.invoke({"messages": [("human", (
            "You are an ML experiment diagnostician. "
            "Call ALL FIVE tools (assess_fitting, assess_convergence, assess_stability, "
            "assess_efficiency, assess_optimization) using these metrics: " + metrics_summary +
            ". After all tool calls, reply with a single JSON object merging all results."
        ))]})
        # Last message content is the final answer
        output = result["messages"][-1].content

        start = output.find("{")
        end   = output.rfind("}") + 1
        data: dict[str, Any] = json.loads(output[start:end])
        return StateSignals(**data)

    except Exception:
        return _rule_based(state)
