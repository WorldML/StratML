"""
state_history.py
----------------
Phase 2 (Dev B) — Trajectory Feature Extraction.
Supports both maximization and minimization optimization directions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Optional

from stratml.execution.schemas import ExperimentResult

_WINDOW = 3


@dataclass
class TrajectoryFeatures:
    history_length: int
    improvement_rate: float
    slope: float
    loss_slope: float
    volatility: float
    best_score: float
    mean_score: float
    steps_since_improvement: int
    runtime_trend: float
    model_switch_frequency: int
    trend: str  # "improving" | "stagnating" | "degrading"


class ExperimentHistory:
    """Rolling buffer of the last N ExperimentResults."""

    def __init__(self, window: int = _WINDOW) -> None:
        self._buffer: deque[ExperimentResult] = deque(maxlen=window)
        self._best_score: Optional[float] = None
        self._steps_since_improvement: int = 0

    def push(self, result: ExperimentResult) -> None:
        self._buffer.append(result)

    def compute_trajectory(
        self,
        primary_metric: str = "accuracy",
        optimization_goal: str = "maximize",
    ) -> TrajectoryFeatures:
        buf = list(self._buffer)
        n = len(buf)

        scores = [float(getattr(r.metrics, primary_metric, None) or 0.0) for r in buf]
        losses = [float(r.metrics.validation_loss or 0.0) for r in buf]
        runtimes = [r.runtime for r in buf]

        current = scores[-1] if scores else 0.0
        prev = scores[-2] if n >= 2 else None

        if prev is not None:
            if optimization_goal == "minimize":
                improvement_rate = round(prev - current, 6)
            else:
                improvement_rate = round(current - prev, 6)
        else:
            improvement_rate = 0.0

        raw_slope = (scores[-1] - scores[0]) / max(n - 1, 1) if n >= 2 else 0.0
        slope = round(raw_slope, 6)
        effective_slope = -slope if optimization_goal == "minimize" else slope

        loss_slope = round((losses[-1] - losses[0]) / max(n - 1, 1), 6) if n >= 2 else 0.0
        runtime_trend = round(runtimes[-1] - runtimes[-2], 4) if n >= 2 else 0.0
        volatility = round(stdev(scores), 6) if n >= 2 else 0.0

        if optimization_goal == "minimize":
            best_score = min(scores) if scores else 0.0
        else:
            best_score = max(scores) if scores else 0.0

        mean_score = round(mean(scores), 6) if scores else 0.0

        if self._best_score is None:
            self._best_score = current
            self._steps_since_improvement = 0
        elif optimization_goal == "minimize" and current < self._best_score - 1e-6:
            self._best_score = current
            self._steps_since_improvement = 0
        elif optimization_goal == "maximize" and current > self._best_score + 1e-6:
            self._best_score = current
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        return TrajectoryFeatures(
            history_length=n,
            improvement_rate=improvement_rate,
            slope=slope,
            loss_slope=loss_slope,
            volatility=volatility,
            best_score=best_score,
            mean_score=mean_score,
            steps_since_improvement=self._steps_since_improvement,
            runtime_trend=runtime_trend,
            model_switch_frequency=_count_model_switches(buf),
            trend=_infer_trend(effective_slope),
        )


def _count_model_switches(buf: list[ExperimentResult]) -> int:
    return sum(1 for i in range(1, len(buf)) if buf[i].model_name != buf[i - 1].model_name)


def _infer_trend(effective_slope: float) -> str:
    if effective_slope > 0.001:
        return "improving"
    if effective_slope < -0.001:
        return "degrading"
    return "stagnating"
