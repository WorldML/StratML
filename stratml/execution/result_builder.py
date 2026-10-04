"""
result_builder.py
-----------------
Phase 8 — Assemble ExperimentResult from all phase outputs.
"""

from __future__ import annotations

from typing import Optional

from stratml.execution.schemas import (
    ExperimentConfig, ExperimentMetrics, ResourceUsage,
    ArtifactRefs, PreprocessingConfig, ExperimentResult,
)


def build_experiment_result(
    config: ExperimentConfig,
    metrics: ExperimentMetrics,
    train_curve: list[float],
    validation_curve: list[float],
    runtime: float,
    resource_usage: ResourceUsage,
    artifacts: ArtifactRefs,
    preprocessing_applied: PreprocessingConfig,
    iteration: int,
    dataset_name: str,
    early_stopped: Optional[bool] = None,
    best_epoch: Optional[int] = None,
    status: Optional[str] = None,
    failed: Optional[bool] = None,
    eval_count: int = 1,
    fit_count: int = 1,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=config.experiment_id,
        iteration=iteration,
        dataset_name=dataset_name,
        model_name=config.model_name,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        preprocessing_applied=preprocessing_applied,
        metrics=metrics,
        train_curve=train_curve,
        validation_curve=validation_curve,
        runtime=runtime,
        resource_usage=resource_usage,
        artifacts=artifacts,
        early_stopped=early_stopped,
        best_epoch=best_epoch,
        status=status,
        failed=failed,
        eval_count=eval_count,
        fit_count=fit_count,
    )
