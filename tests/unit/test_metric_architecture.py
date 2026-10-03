"""
test_metric_architecture.py
---------------------------
Unit tests for Phase 1: Metric Architecture + Optimization Direction.
Verifies end-to-end integration of ROC-AUC, Log Loss, MAE, RMSE,
minimization direction semantics, trajectory tracking, and native ARFF loading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stratml.core.schemas import ExperimentMetrics, PreprocessingConfig
from stratml.core.metrics import (
    resolve_canonical_metric,
    resolve_task_type,
    is_better_score,
    compute_semantic_gain,
)
from stratml.execution.metrics.metrics_engine import compute_metrics
from stratml.execution.schemas import ExperimentConfig, DataSplit
from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline
from stratml.execution.data.loader import load_dataframe
from stratml.execution.data.validator import build_dataset
from stratml.execution.data.profiler import build_profile
from stratml.decision.state.state_history import ExperimentHistory
from stratml.decision.state.signals import assess_convergence, assess_fitting, _rule_based
from stratml.decision.state.state_builder import build_state
from stratml.core.schemas import ExperimentResult, ResourceUsage, ArtifactRefs


class TestCanonicalMetricResolution:
    def test_binary_classification_defaults(self):
        primary, goal, secondary = resolve_canonical_metric("classification", n_classes=2)
        assert primary == "roc_auc"
        assert goal == "maximize"
        assert "f1_score" in secondary

    def test_multiclass_classification_defaults(self):
        primary, goal, secondary = resolve_canonical_metric("classification", n_classes=4)
        assert primary == "log_loss"
        assert goal == "minimize"
        assert "f1_score" in secondary

    def test_regression_defaults(self):
        primary, goal, secondary = resolve_canonical_metric("regression")
        assert primary == "rmse"
        assert goal == "minimize"
        assert "mae" in secondary
        assert "r2" in secondary

    def test_explicit_metric_overrides(self):
        primary, goal, _ = resolve_canonical_metric("regression", explicit_metric="r2")
        assert primary == "r2"
        assert goal == "maximize"

        primary_acc, goal_acc, _ = resolve_canonical_metric("classification", n_classes=2, explicit_metric="accuracy")
        assert primary_acc == "accuracy"
        assert goal_acc == "maximize"


class TestMetricsEngineComputation:
    def test_binary_roc_auc_with_probabilities(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        # Perfect probabilities
        y_proba = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.05, 0.95],
            [0.85, 0.15],
        ])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            train_curve=[0.2],
            val_curve=[0.2],
            problem_type="classification",
            y_proba=y_proba,
            classes=[0, 1],
            task_type="binary_classification",
        )
        assert m.roc_auc == 1.0
        assert m.log_loss is not None
        assert m.log_loss < 0.5
        assert m.accuracy == 1.0
        assert m.f1_score == 1.0

    def test_binary_roc_auc_with_string_labels(self):
        y_true = np.array(["bad", "good", "bad", "good"])
        y_proba = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.2, 0.8],
        ])
        y_pred = np.array(["bad", "good", "bad", "good"])
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            train_curve=[0.1],
            val_curve=[0.1],
            problem_type="classification",
            y_proba=y_proba,
            classes=["bad", "good"],
            task_type="binary_classification",
        )
        assert m.roc_auc == 1.0
        assert m.log_loss is not None

    def test_multiclass_log_loss_computation(self):
        y_true = np.array([0, 1, 2, 1])
        y_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.05, 0.9, 0.05],
        ])
        y_pred = np.array([0, 1, 2, 1])
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            train_curve=[0.3],
            val_curve=[0.3],
            problem_type="classification",
            y_proba=y_proba,
            classes=[0, 1, 2],
            task_type="multiclass_classification",
        )
        assert m.log_loss is not None
        assert m.log_loss > 0.0
        assert m.f1_score == 1.0

    def test_regression_rmse_mae_r2(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 31.0, 39.0])
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            train_curve=[1.5],
            val_curve=[1.5],
            problem_type="regression",
        )
        assert m.mae == 1.5
        assert m.rmse == pytest.approx(1.581139, rel=1e-3)
        assert m.r2 > 0.95

    def test_edge_case_single_class_batch_does_not_crash(self):
        y_true = np.array([1, 1, 1])
        y_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.1, 0.9]])
        y_pred = np.array([1, 1, 1])
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            train_curve=[],
            val_curve=[],
            problem_type="classification",
            y_proba=y_proba,
            classes=[0, 1],
        )
        assert m.roc_auc == 0.5  # Neutral fallback for single class batch
        assert m.accuracy == 1.0


class TestOptimizationDirectionAndSemanticGain:
    def test_semantic_gain_minimization(self):
        # When minimizing loss/rmse, reducing from 10.0 to 7.0 is a positive improvement of 3.0
        gain = compute_semantic_gain(current=7.0, baseline=10.0, optimization_goal="minimize")
        assert gain == 3.0

        # Degradation from 7.0 to 9.0 is negative
        deg_gain = compute_semantic_gain(current=9.0, baseline=7.0, optimization_goal="minimize")
        assert deg_gain == -2.0

    def test_semantic_gain_maximization(self):
        # When maximizing ROC-AUC, increasing from 0.80 to 0.85 is a positive improvement of 0.05
        gain = compute_semantic_gain(current=0.85, baseline=0.80, optimization_goal="maximize")
        assert gain == 0.05

    def test_is_better_score(self):
        assert is_better_score(current=5.0, baseline=6.0, optimization_goal="minimize") is True
        assert is_better_score(current=7.0, baseline=6.0, optimization_goal="minimize") is False
        assert is_better_score(current=0.9, baseline=0.8, optimization_goal="maximize") is True
        assert is_better_score(current=0.7, baseline=0.8, optimization_goal="maximize") is False


class TestTrajectoryTrackingUnderMinimization:
    def test_history_trajectory_under_rmse_minimization(self):
        history = ExperimentHistory(window=5)

        def make_result(iteration: int, rmse: float):
            return ExperimentResult(
                experiment_id=f"exp_{iteration}",
                iteration=iteration,
                model_name="RandomForestRegressor",
                model_type="ml",
                hyperparameters={},
                preprocessing_applied=PreprocessingConfig(),
                metrics=ExperimentMetrics(rmse=rmse, validation_loss=rmse),
                train_curve=[rmse],
                validation_curve=[rmse],
                runtime=1.0,
                resource_usage=ResourceUsage(),
                artifacts=ArtifactRefs(),
            )

        # Iteration 1: RMSE = 10.0
        history.push(make_result(1, 10.0))
        traj1 = history.compute_trajectory(primary_metric="rmse", optimization_goal="minimize")
        assert traj1.best_score == 10.0
        assert traj1.steps_since_improvement == 0

        # Iteration 2: RMSE = 8.0 (improving!)
        history.push(make_result(2, 8.0))
        traj2 = history.compute_trajectory(primary_metric="rmse", optimization_goal="minimize")
        assert traj2.best_score == 8.0  # min score is best for minimization!
        assert traj2.improvement_rate == 2.0  # positive improvement rate!
        assert traj2.trend == "improving"
        assert traj2.steps_since_improvement == 0

        # Iteration 3: RMSE = 9.0 (degraded)
        history.push(make_result(3, 9.0))
        traj3 = history.compute_trajectory(primary_metric="rmse", optimization_goal="minimize")
        assert traj3.best_score == 8.0
        assert traj3.steps_since_improvement == 1


class TestSignalExtractionMinimization:
    def test_diverging_signal_under_minimization(self):
        # When minimizing, a positive slope (e.g. +0.03 error increase) means diverging
        fn = getattr(assess_convergence, "func", assess_convergence)
        res = fn(slope=0.03, primary=1.5, steps_since=1, goal="minimize")
        import json
        data = json.loads(res)
        assert data["diverging"] == "strong"

    def test_converged_signal_under_minimization(self):
        fn = getattr(assess_convergence, "func", assess_convergence)
        res = fn(slope=0.0002, primary=0.20, steps_since=0, goal="minimize")
        import json
        data = json.loads(res)
        assert data["converged"] == "strong"



class TestMLPipelineProbabilities:
    def test_svc_automatically_enables_predict_proba(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        split = DataSplit(
            X_train=X.iloc[:4], X_val=X.iloc[4:], X_test=X.iloc[4:],
            y_train=y.iloc[:4], y_val=y.iloc[4:], y_test=y.iloc[4:],
        )
        cfg = ExperimentConfig(
            experiment_id="exp_svc",
            model_name="SVC",
            model_type="ml",
            hyperparameters={"C": 1.0},
            preprocessing=PreprocessingConfig(),
        )
        pipe = run_ml_pipeline(cfg, split)
        assert pipe.y_val_proba is not None
        assert pipe.y_val_proba.shape == (2, 2)
        assert pipe.classes is not None


class TestNativeARFFLoading:
    def test_load_binary_arff_australian(self):
        path = "data/research/classification/australian.arff"
        if not Path(path).exists():
            pytest.skip(f"Benchmark file {path} not found")
        df, name = load_dataframe(path)
        assert name == "australian"
        assert len(df) == 690
        dataset = build_dataset(df, name)
        assert dataset.target_column == "A15"
        profile = build_profile(dataset)
        assert profile.problem_type == "classification"
        assert len(profile.class_distribution) == 2
        assert profile.recommended_metrics[0] == "roc_auc"

    def test_load_multiclass_arff_vehicle(self):
        path = "data/research/classification/vehicle.arff"
        if not Path(path).exists():
            pytest.skip(f"Benchmark file {path} not found")
        df, name = load_dataframe(path)
        assert name == "vehicle"
        assert len(df) == 846
        dataset = build_dataset(df, name)
        assert dataset.target_column == "Class"
        profile = build_profile(dataset)
        assert profile.problem_type == "classification"
        assert len(profile.class_distribution) == 4
        assert profile.recommended_metrics[0] == "log_loss"

    def test_load_regression_arff_boston(self):
        path = "data/research/regression/boston.arff"
        if not Path(path).exists():
            pytest.skip(f"Benchmark file {path} not found")
        df, name = load_dataframe(path)
        assert name == "boston"
        assert len(df) == 506
        dataset = build_dataset(df, name)
        assert dataset.target_column == "MEDV"
        profile = build_profile(dataset)
        assert profile.problem_type == "regression"
        assert profile.recommended_metrics[0] == "rmse"
        assert "mae" in profile.recommended_metrics
        assert "r2" in profile.recommended_metrics
