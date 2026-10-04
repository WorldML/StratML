"""
tests/unit/test_random_search.py
--------------------------------
Comprehensive unit and smoke test suite for the Random Search Baseline under Phase 3.

Verifies:
- Test A: Deterministic seed produces identical candidate sequence
- Test B: Budget ceiling of 10 model evaluations
- Test C: Budget ceiling of 20 model evaluations
- Test D: Metric parity across binary, multiclass, and regression tasks
- Test E: Directionality handling (maximize vs minimize)
- Test F: Failed evaluation accounting without retry loop
- Test G: Duplicate configuration execution and accounting
- Test H: Manifest structure and required fields
- Test I: Test-set isolation (test set unused during search)
- Test J: End-to-end reproducibility with identical seeds
- Smoke Tests: Australian (binary), Vehicle (multiclass), Boston (regression)
- Unified Runner: run_experiment() interface
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stratml.baselines.random_search import (
    RANDOM_SEARCH_CLASSIFICATION_MODELS,
    RANDOM_SEARCH_PARAM_SPACES,
    RANDOM_SEARCH_REGRESSION_MODELS,
    RandomSearchEngine,
    RandomSearchSampler,
)
from stratml.core.metrics import is_better_score, resolve_canonical_metric
from stratml.core.schemas import CanonicalExperimentResult
from stratml.execution.schemas import (
    DataProfile,
    ExperimentMetrics,
    ExperimentResult,
    FeatureInfo,
    PreprocessingConfig,
    ResourceUsage,
    ArtifactRefs,
    SplitConfig,
)
from stratml.orchestration.orchestrator import ExecutionOrchestrator
from stratml.runner import run_experiment


def _make_dummy_dataset(
    tmp_path: Path,
    n_samples: int = 100,
    task: str = "classification",
    n_classes: int = 2,
) -> tuple[Path, str]:
    """Helper to generate a clean tabular dataset for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, 4)
    if task == "classification":
        if n_classes == 2:
            y = rng.choice([0, 1], size=n_samples)
        else:
            y = rng.choice(list(range(n_classes)), size=n_samples)
    else:
        y = rng.randn(n_samples)

    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["target"] = y
    csv_path = tmp_path / f"dummy_{task}_{n_classes}c.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, "target"


class TestRandomSearchBasics:
    """Core requirements tests A through J for Random Search."""

    def test_a_deterministic_seed(self):
        """Test A: Same seed + same configuration -> same candidate sequence."""
        sampler1 = RandomSearchSampler(seed=42)
        sampler2 = RandomSearchSampler(seed=42)
        sampler3 = RandomSearchSampler(seed=999)

        seq1 = [sampler1.sample_candidate("classification") for _ in range(10)]
        seq2 = [sampler2.sample_candidate("classification") for _ in range(10)]
        seq3 = [sampler3.sample_candidate("classification") for _ in range(10)]

        assert seq1 == seq2, "Identical seeds must produce identical candidate sequences"
        assert seq1 != seq3, "Different seeds must produce different candidate sequences"

    def test_b_budget_10(self, tmp_path):
        """Test B: When budget=10, exactly 10 candidate evaluations complete."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id = f"test_b_{uuid.uuid4().hex[:6]}"

        engine = RandomSearchEngine(
            evaluation_budget=10,
            seed=42,
            run_id=run_id,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=10,
            tune=False,
            system="random_search",
        )

        try:
            result = orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 10
            assert orchestrator.actual_fits == 10
            assert orchestrator.termination_reason == "budget_exhausted"
            assert isinstance(result, CanonicalExperimentResult)
            assert result.actual_evaluations == 10
            assert len(result.trajectory) == 10
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_c_budget_20(self, tmp_path):
        """Test C: When budget=20, exactly 20 candidate evaluations complete."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id = f"test_c_{uuid.uuid4().hex[:6]}"

        engine = RandomSearchEngine(
            evaluation_budget=20,
            seed=42,
            run_id=run_id,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=20,
            tune=False,
            system="random_search",
        )

        try:
            result = orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 20
            assert orchestrator.actual_fits == 20
            assert orchestrator.termination_reason == "budget_exhausted"
            assert result.actual_evaluations == 20
            assert len(result.trajectory) == 20
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_d_metric_parity(self):
        """Test D: Metric resolver parity across binary, multiclass, and regression."""
        # Binary classification
        m_bin, g_bin, s_bin = resolve_canonical_metric("classification", 2)
        assert m_bin == "roc_auc"
        assert g_bin == "maximize"
        assert "f1_score" in s_bin

        # Multiclass classification
        m_mc, g_mc, s_mc = resolve_canonical_metric("classification", 5)
        assert m_mc == "log_loss"
        assert g_mc == "minimize"
        assert "f1_score" in s_mc

        # Regression
        m_reg, g_reg, s_reg = resolve_canonical_metric("regression", None)
        assert m_reg == "rmse"
        assert g_reg == "minimize"
        assert "mae" in s_reg
        assert "r2" in s_reg

    def test_e_directionality(self):
        """Test E: is_better_score handles maximize and minimize metrics without hardcoded assumptions."""
        # Maximization (ROC-AUC, F1, R2)
        assert is_better_score(0.85, 0.70, "maximize") is True
        assert is_better_score(0.65, 0.70, "maximize") is False

        # Minimization (Log Loss, RMSE, MAE)
        assert is_better_score(0.15, 0.30, "minimize") is True
        assert is_better_score(0.45, 0.30, "minimize") is False

        # RandomSearchEngine best_val_score tracking under minimization
        profile = DataProfile(
            dataset_name="dummy",
            dataset_type="tabular",
            rows=100,
            columns=5,
            target_column="target",
            problem_type="regression",
            numerical_columns=["f0", "f1", "f2", "f3"],
            categorical_columns=[],
            missing_value_ratio=0.0,
            feature_summary=[],
            recommended_metrics=["rmse"],
        )
        engine = RandomSearchEngine(evaluation_budget=5, seed=42)
        engine.receive_profile(profile)
        assert engine.primary_metric == "rmse"
        assert engine.optimization_goal == "minimize"

        # Simulate reporting scores: 0.50 then 0.30 then 0.40
        dummy_res1 = ExperimentResult(
            experiment_id="exp1",
            iteration=1,
            dataset_name="dummy",
            model_name="Ridge",
            model_type="ml",
            hyperparameters={"alpha": 1.0},
            preprocessing_applied=PreprocessingConfig(),
            metrics=ExperimentMetrics(rmse=0.50),
            train_curve=[],
            validation_curve=[],
            runtime=0.1,
            resource_usage=ResourceUsage(),
            artifacts=ArtifactRefs(model_path="", metrics_file="", tensorboard_logs=""),
        )
        engine.receive_result(dummy_res1)
        assert engine.best_val_score == 0.50

        dummy_res2 = ExperimentResult(
            experiment_id="exp2",
            iteration=2,
            dataset_name="dummy",
            model_name="Ridge",
            model_type="ml",
            hyperparameters={"alpha": 0.1},
            preprocessing_applied=PreprocessingConfig(),
            metrics=ExperimentMetrics(rmse=0.30),
            train_curve=[],
            validation_curve=[],
            runtime=0.1,
            resource_usage=ResourceUsage(),
            artifacts=ArtifactRefs(model_path="", metrics_file="", tensorboard_logs=""),
        )
        engine.receive_result(dummy_res2)
        assert engine.best_val_score == 0.30  # lower is better

        dummy_res3 = ExperimentResult(
            experiment_id="exp3",
            iteration=3,
            dataset_name="dummy",
            model_name="Ridge",
            model_type="ml",
            hyperparameters={"alpha": 10.0},
            preprocessing_applied=PreprocessingConfig(),
            metrics=ExperimentMetrics(rmse=0.40),
            train_curve=[],
            validation_curve=[],
            runtime=0.1,
            resource_usage=ResourceUsage(),
            artifacts=ArtifactRefs(model_path="", metrics_file="", tensorboard_logs=""),
        )
        engine.receive_result(dummy_res3)
        assert engine.best_val_score == 0.30  # 0.40 does not overwrite 0.30

    def test_f_failure(self, tmp_path):
        """Test F: Failed candidate consumes 1 evaluation attempt and avoids infinite retry loop."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id = f"test_f_{uuid.uuid4().hex[:6]}"

        engine = RandomSearchEngine(
            evaluation_budget=3,
            seed=42,
            run_id=run_id,
            allowed_models=["LogisticRegression"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=3,
            tune=False,
            system="random_search",
        )

        with patch("stratml.orchestration.orchestrator.run_ml_pipeline", side_effect=RuntimeError("Pipeline crash")):
            try:
                result = orchestrator.run(str(csv_path), target)
                assert orchestrator.actual_evaluations == 3
                assert orchestrator.actual_fits == 3
                assert orchestrator.termination_reason == "budget_exhausted"
                assert result.actual_evaluations == 3
                assert all(step.status == "failed" for step in result.trajectory)
            finally:
                shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_g_duplicate(self, tmp_path):
        """Test G: Duplicate configuration execution consumes 1 evaluation and increments repeated_configs."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id = f"test_g_{uuid.uuid4().hex[:6]}"

        # Force sampler to sample a single fixed configuration
        fixed_spaces = {"LogisticRegression": {"C": [1.0], "solver": ["lbfgs"]}}
        engine = RandomSearchEngine(
            evaluation_budget=4,
            seed=42,
            run_id=run_id,
            allowed_models=["LogisticRegression"],
            param_spaces=fixed_spaces,
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=4,
            tune=False,
            system="random_search",
        )

        try:
            result = orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 4
            assert orchestrator.repeated_configs == 3  # 1st unique, 3 duplicates
            assert engine.repeated_configs == 3
            assert result.manifest["budget"]["actual_consumption"]["repeated_configs"] == 3
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_h_manifest(self, tmp_path):
        """Test H: Manifest emits all required fields with exact provenance."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id = f"test_h_{uuid.uuid4().hex[:6]}"

        engine = RandomSearchEngine(
            evaluation_budget=3,
            seed=42,
            run_id=run_id,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=3,
            tune=False,
            system="random_search",
        )

        try:
            result = orchestrator.run(str(csv_path), target)
            manifest = result.manifest

            # Mandatory fields verification
            assert manifest["system"] == "random_search"
            assert "dataset" in manifest
            assert manifest["task_type"] == "binary_classification"
            assert manifest["seed"] == 42
            assert manifest["evaluation_budget"] == 3
            assert manifest["actual_evaluations"] == 3
            assert manifest["actual_fits"] == 3
            assert manifest["decision_iterations"] == 3
            assert manifest["termination_reason"] == "budget_exhausted"
            assert manifest["best_validation_score"] is not None
            assert manifest["best_test_score"] is not None
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_i_test_isolation(self, tmp_path):
        """Test I: Verify test data is strictly isolated during search and only evaluated at finalization."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id = f"test_i_{uuid.uuid4().hex[:6]}"

        call_records = []

        def recording_compute_metrics(*args, **kwargs):
            from stratml.execution.metrics.metrics_engine import compute_metrics
            call_records.append(len(kwargs.get("y_true", [])))
            return compute_metrics(*args, **kwargs)

        engine = RandomSearchEngine(
            evaluation_budget=3,
            seed=42,
            run_id=run_id,
            allowed_models=["LogisticRegression"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=3,
            tune=False,
            system="random_search",
        )

        with patch("stratml.orchestration.orchestrator.compute_metrics", side_effect=recording_compute_metrics):
            try:
                orchestrator.run(str(csv_path), target)
                # In standard split: 80 total rows, test_size=0.2 (16 rows), train+val=64 rows, val_size=0.1 of 64 (~7 rows)
                # First 3 calls are for validation split; final call is for test split
                assert len(call_records) == 4
                val_sample_counts = call_records[:3]
                test_sample_count = call_records[3]
                assert val_sample_counts[0] == val_sample_counts[1] == val_sample_counts[2]
                assert test_sample_count != val_sample_counts[0]
                assert test_sample_count == 16, "Final call must evaluate on the 16 test samples"
            finally:
                shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_j_reproducibility(self, tmp_path):
        """Test J: Two complete runs with identical seed and configuration produce identical sequences."""
        csv_path, target = _make_dummy_dataset(tmp_path, n_samples=80)
        run_id_1 = f"test_j1_{uuid.uuid4().hex[:6]}"
        run_id_2 = f"test_j2_{uuid.uuid4().hex[:6]}"

        res1 = run_experiment(
            str(csv_path),
            target,
            system="random_search",
            budget=5,
            seed=123,
            run_id=run_id_1,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )
        res2 = run_experiment(
            str(csv_path),
            target,
            system="random_search",
            budget=5,
            seed=123,
            run_id=run_id_2,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )

        try:
            seq1 = [(s.model_name, s.hyperparameters) for s in res1.trajectory]
            seq2 = [(s.model_name, s.hyperparameters) for s in res2.trajectory]
            assert seq1 == seq2, "Candidate sequences must be identical across runs with identical seed"
            assert res1.best_validation_score == res2.best_validation_score
            assert res1.best_test_score == res2.best_test_score
        finally:
            shutil.rmtree(Path("outputs") / run_id_1, ignore_errors=True)
            shutil.rmtree(Path("outputs") / run_id_2, ignore_errors=True)


class TestRandomSearchSmokeARFF:
    """Smoke tests on frozen representative research datasets with budget=2."""

    def test_smoke_australian_binary_classification(self):
        """Smoke test binary classification on australian.arff."""
        arff_path = Path("data/research/classification/australian.arff")
        if not arff_path.exists():
            pytest.skip("australian.arff not found in data/research/classification/")

        run_id = f"smoke_rs_australian_{uuid.uuid4().hex[:6]}"
        try:
            res = run_experiment(
                dataset_path=str(arff_path),
                target_column="A15",
                system="random_search",
                budget=2,
                seed=42,
                run_id=run_id,
            )
            assert res.system == "random_search"
            assert res.actual_evaluations == 2
            assert res.actual_fits == 2
            assert res.task_type == "binary_classification"
            assert res.manifest["evaluation_configuration"]["primary_metric"] == "roc_auc"
            assert res.manifest["evaluation_configuration"]["optimization_goal"] == "maximize"
            assert res.best_validation_score is not None
            assert res.best_test_score is not None
            assert res.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_smoke_vehicle_multiclass_classification(self):
        """Smoke test multiclass classification on vehicle.arff."""
        arff_path = Path("data/research/classification/vehicle.arff")
        if not arff_path.exists():
            pytest.skip("vehicle.arff not found in data/research/classification/")

        run_id = f"smoke_rs_vehicle_{uuid.uuid4().hex[:6]}"
        try:
            res = run_experiment(
                dataset_path=str(arff_path),
                target_column="Class",
                system="random_search",
                budget=2,
                seed=42,
                run_id=run_id,
            )
            assert res.system == "random_search"
            assert res.actual_evaluations == 2
            assert res.actual_fits == 2
            assert res.task_type == "multiclass_classification"
            assert res.manifest["evaluation_configuration"]["primary_metric"] == "log_loss"
            assert res.manifest["evaluation_configuration"]["optimization_goal"] == "minimize"
            assert res.best_validation_score is not None
            assert res.best_test_score is not None
            assert res.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_smoke_boston_regression(self):
        """Smoke test regression on boston.arff."""
        arff_path = Path("data/research/regression/boston.arff")
        if not arff_path.exists():
            pytest.skip("boston.arff not found in data/research/regression/")

        run_id = f"smoke_rs_boston_{uuid.uuid4().hex[:6]}"
        try:
            res = run_experiment(
                dataset_path=str(arff_path),
                target_column="MEDV",
                system="random_search",
                budget=2,
                seed=42,
                run_id=run_id,
            )
            assert res.system == "random_search"
            assert res.actual_evaluations == 2
            assert res.actual_fits == 2
            assert res.task_type == "regression"
            assert res.manifest["evaluation_configuration"]["primary_metric"] == "rmse"
            assert res.manifest["evaluation_configuration"]["optimization_goal"] == "minimize"
            assert res.best_validation_score is not None
            assert res.best_test_score is not None
            assert res.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)
