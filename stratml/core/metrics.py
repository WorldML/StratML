"""
metrics.py
----------
Canonical metric and optimization direction resolution for StratML.

Frozen Paper Protocol ("Beyond Configuration Search: A State-Aware Framework for AutoML Experimentation"):
- Binary classification:
    primary:   roc_auc (maximize)
    secondary: [f1_score]
- Multiclass classification:
    primary:   log_loss (minimize)
    secondary: [f1_score]
- Regression:
    primary:   rmse (minimize)
    secondary: [mae, r2]
"""

from __future__ import annotations

from typing import Optional, Sequence


def resolve_task_type(problem_type: str, n_classes: Optional[int] = None) -> str:
    """
    Resolve problem_type and number of classes to canonical task type:
    - 'binary_classification'
    - 'multiclass_classification'
    - 'regression'
    """
    if str(problem_type).lower() == "regression":
        return "regression"
    if n_classes is not None and n_classes > 2:
        return "multiclass_classification"
    return "binary_classification"


def resolve_canonical_metric(
    problem_type: str,
    n_classes: Optional[int] = None,
    explicit_metric: Optional[str] = None,
    explicit_goal: Optional[str] = None,
) -> tuple[str, str, list[str]]:
    """
    Resolve primary metric, optimization goal ("maximize" | "minimize"), and secondary metrics.

    Priority:
    1. If explicit_metric is specified, respect it.
    2. Otherwise, use frozen protocol defaults.

    Returns:
        (primary_metric, optimization_goal, secondary_metrics)
    """
    task = resolve_task_type(problem_type, n_classes)

    if task == "regression":
        primary = explicit_metric or "rmse"
        if explicit_goal is not None:
            goal = explicit_goal
        elif primary == "r2":
            goal = "maximize"
        else:
            goal = "minimize"
        secondary = ["mae", "r2"]

    elif task == "multiclass_classification":
        primary = explicit_metric or "log_loss"
        if explicit_goal is not None:
            goal = explicit_goal
        elif primary in ("accuracy", "f1_score", "roc_auc"):
            goal = "maximize"
        else:
            goal = "minimize"
        secondary = ["f1_score"]

    else:  # binary_classification
        primary = explicit_metric or "roc_auc"
        if explicit_goal is not None:
            goal = explicit_goal
        elif primary == "log_loss":
            goal = "minimize"
        else:
            goal = "maximize"
        secondary = ["f1_score"]

    return primary, goal, secondary


def is_better_score(current: float, baseline: float, optimization_goal: str) -> bool:
    """Check if current score is strictly better than baseline under given goal."""
    if optimization_goal == "minimize":
        return current < baseline - 1e-6
    return current > baseline + 1e-6


def compute_semantic_gain(
    current: float,
    baseline: Optional[float],
    optimization_goal: str,
) -> float:
    """
    Compute semantic gain such that gain > 0 ALWAYS means improvement.

    - For maximize: delta = current - baseline
    - For minimize: delta = baseline - current (loss reduction)
    """
    if baseline is None:
        return 0.0
    if optimization_goal == "minimize":
        return round(float(baseline - current), 6)
    return round(float(current - baseline), 6)
