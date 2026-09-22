"""
state_builder.py
----------------
Dev B — State Pipeline orchestrator.

Two public entry points:
    build_state_from_profile(profile, history)  — Iteration 0
    build_state(result, history, profile)        — Iteration 1+
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from stratml.execution.schemas import DataProfile, ExperimentResult
from stratml.core.schemas import (
    StateObject,
    StateMeta,
    StateObjective,
    StateMetrics,
    SecondaryMetrics,
    StateTrajectory,
    StateDataset,
    StateModel,
    StateGeneralization,
    StateResources,
    StateSearch,
    StateSignals,
    StateUncertainty,
    StateActionContext,
    StateConstraints,
)
from stratml.decision.state.state_history import ExperimentHistory
from stratml.decision.state.meta_features import extract as extract_meta
from stratml.decision.state.signals import compute_signals


def build_state_from_profile(
    profile: DataProfile,
    *,
    run_id: str = "run",
    primary_metric: str = "accuracy",
    optimization_goal: str = "maximize",
    allowed_models: Optional[list[str]] = None,
    max_iterations: int = 20,
    time_budget: Optional[float] = None,
) -> StateObject:
    """Iteration 0 entry point — bootstrap StateObject from DataProfile."""
    num_samples = profile.rows
    num_features = profile.columns - 1

    class_dist = dict(profile.class_distribution) if profile.class_distribution else None
    imbalance_ratio: Optional[float] = None
    if class_dist and len(class_dist) >= 2:
        counts = list(class_dist.values())
        imbalance_ratio = round(max(counts) / max(min(counts), 1), 4)

    return StateObject(
        meta=StateMeta(
            experiment_id=run_id,
            iteration=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        objective=StateObjective(
            primary_metric=primary_metric,
            optimization_goal=optimization_goal,
        ),
        metrics=StateMetrics(primary=0.0, secondary=SecondaryMetrics(), train_val_gap=0.0),
        trajectory=StateTrajectory(
            history_length=0, improvement_rate=0.0, slope=0.0, volatility=0.0,
            best_score=0.0, mean_score=0.0, steps_since_improvement=0, trend="stagnating",
        ),
        dataset=StateDataset(
            num_samples=num_samples,
            num_features=num_features,
            feature_to_sample_ratio=round(num_features / max(num_samples, 1), 6),
            missing_ratio=profile.missing_value_ratio,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
            dataset_fingerprint=getattr(profile, "dataset_fingerprint", None),
        ),
        model=StateModel(
            model_name="none", model_type="ml", hyperparameters={},
            complexity_hint=None, runtime=0.0, convergence_epoch=0, early_stopped=None,
        ),
        generalization=StateGeneralization(train_loss=0.0, validation_loss=0.0, gap=0.0),
        resources=StateResources(
            runtime=0.0, gpu_used=False, cpu_time=0.0,
            remaining_budget=float(max_iterations), budget_exhausted=False,
        ),
        search=StateSearch(models_tried=[], unique_models_count=0, repeated_configs=0),
        signals=StateSignals(),
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(),
        constraints=StateConstraints(
            allowed_models=allowed_models or [],
            max_iterations=max_iterations,
            time_budget=time_budget,
        ),
    )


def build_state(
    result: ExperimentResult,
    *,
    history: Optional[ExperimentHistory] = None,
    profile: Optional[DataProfile] = None,
    primary_metric: str = "accuracy",
    optimization_goal: str = "maximize",
    allowed_models: Optional[list[str]] = None,
    max_iterations: int = 20,
    time_budget: Optional[float] = None,
    previous_action: Optional[str] = None,
    previous_action_success: Optional[bool] = None,
    models_tried: Optional[list[str]] = None,
    repeated_configs: int = 0,
    remaining_budget: Optional[float] = None,
    previous_signals=None,
    execution_status: Optional[str] = "completed",
    action_outcome: Optional[str] = None,
) -> StateObject:
    """Iteration 1+ entry point — full pipeline from ExperimentResult."""
    if history is None:
        history = ExperimentHistory()
    history.push(result)
    if profile is not None and getattr(profile, "problem_type", None) == "regression" and primary_metric == "accuracy":
        primary_metric = "r2"
    traj = history.compute_trajectory(primary_metric)

    improvement_rate = traj.improvement_rate
    if optimization_goal == "minimize":
        improvement_rate = -improvement_rate

    if profile is not None:
        meta = extract_meta(profile)
        num_samples = meta.num_samples
        num_features = meta.num_features
        fsr = meta.feature_sample_ratio
        missing_ratio = meta.missing_value_ratio
        class_dist = dict(profile.class_distribution) if profile.class_distribution else None
        imbalance_ratio = meta.imbalance_ratio
    else:
        num_samples = num_features = 0
        fsr = missing_ratio = 0.0
        class_dist = None
        imbalance_ratio = None

    train_loss = result.metrics.train_loss or 0.0
    val_loss = result.metrics.validation_loss or 0.0
    gap = round(val_loss - train_loss, 6)

    models_tried = list(models_tried or [result.model_name])
    if result.model_name not in models_tried:
        models_tried.append(result.model_name)

    state = StateObject(
        meta=StateMeta(
            experiment_id=result.experiment_id,
            iteration=result.iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        objective=StateObjective(primary_metric=primary_metric, optimization_goal=optimization_goal),
        metrics=StateMetrics(
            primary=traj.best_score if optimization_goal == "maximize" else traj.mean_score,
            secondary=SecondaryMetrics(
                accuracy=result.metrics.accuracy,
                precision=result.metrics.precision,
                recall=result.metrics.recall,
                f1_score=result.metrics.f1_score,
                mse=result.metrics.mse,
                rmse=result.metrics.rmse,
                r2=result.metrics.r2,
            ),
            train_val_gap=gap,
        ),
        trajectory=StateTrajectory(
            history_length=traj.history_length,
            improvement_rate=improvement_rate,
            slope=traj.slope,
            volatility=traj.volatility,
            best_score=traj.best_score,
            mean_score=traj.mean_score,
            steps_since_improvement=traj.steps_since_improvement,
            trend=traj.trend,
        ),
        dataset=StateDataset(
            num_samples=num_samples,
            num_features=num_features,
            feature_to_sample_ratio=fsr,
            missing_ratio=missing_ratio,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
            dataset_fingerprint=getattr(profile, "dataset_fingerprint", None) if profile is not None else None,
        ),
        model=StateModel(
            model_name=result.model_name,
            model_type=result.model_type,
            hyperparameters=result.hyperparameters,
            complexity_hint=_infer_complexity(result.hyperparameters),
            runtime=result.runtime,
            convergence_epoch=result.best_epoch if result.best_epoch is not None else len(result.train_curve),
            early_stopped=result.early_stopped,
        ),
        generalization=StateGeneralization(train_loss=train_loss, validation_loss=val_loss, gap=gap),
        resources=StateResources(
            runtime=result.runtime,
            gpu_used=result.resource_usage.gpu_used,
            cpu_time=result.resource_usage.cpu_time_sec,
            remaining_budget=remaining_budget,
            budget_exhausted=(remaining_budget is not None and remaining_budget <= 0),
        ),
        search=StateSearch(
            models_tried=models_tried,
            unique_models_count=len(set(models_tried)),
            repeated_configs=repeated_configs,
        ),
        signals=StateSignals(),
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(
            previous_action=previous_action,
            previous_action_success=previous_action_success,
            action_effect_magnitude=abs(improvement_rate) if improvement_rate else None,
            previous_signals=previous_signals,
            execution_status=execution_status,
            action_outcome=action_outcome,
        ),
        constraints=StateConstraints(
            allowed_models=allowed_models or [],
            max_iterations=max_iterations,
            time_budget=time_budget,
        ),
    )

    state.signals = compute_signals(state)
    return state


def _infer_complexity(hyperparameters: dict) -> Optional[str]:
    n = hyperparameters.get("n_estimators") or hyperparameters.get("hidden_units") or 0
    layers = hyperparameters.get("num_layers") or hyperparameters.get("layers") or 1
    depth = hyperparameters.get("max_depth") or 0
    score = int(n) // 100 + int(layers) + int(depth) // 5
    if score == 0:
        return None
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    return "high"
