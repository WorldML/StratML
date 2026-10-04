"""
test_budget_enforcement.py
--------------------------
Phase 2 Unit Tests — Budget Semantics and Enforcement.

Covers:
- Test A: Budget = 10 ceiling never exceeded
- Test B: Budget = 20 ceiling never exceeded
- Test C: Early termination (agent terminates before budget ceiling)
- Test D: Budget exhaustion (terminates at exactly configured ceiling)
- Test E: Duplicate configuration accounting
- Test F: Failed evaluation accounting and prevention of infinite retry loop
- Test G: tune=False accounting (evaluations == fits == iterations)
- Test H: tune=True accounting (model evaluations vs model fits)
- Test I: Decision iterations != evaluation count
- Test J: Manifest budget records (configured budget, actual evaluations, fits, termination reason)
- Dataset Smoke Tests: binary (australian), multiclass (vehicle), regression (boston)
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

from stratml.orchestration.orchestrator import ExecutionOrchestrator
from stratml.execution.schemas import (
    ActionDecision, ExperimentConfig, ExperimentResult, PreprocessingConfig,
    DataProfile, SplitConfig,
)
from stratml.decision.engine import DecisionEngine


def _make_dummy_dataset(tmp_path: Path, n_samples: int = 100, task: str = "classification") -> tuple[Path, str]:
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, 4)
    if task == "classification":
        y = rng.choice([0, 1], size=n_samples)
    else:
        y = rng.randn(n_samples)

    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["target"] = y
    csv_path = tmp_path / f"dummy_{task}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, "target"


def _make_stub_action(model_name: str = "LogisticRegression", action_type: str = "switch_model", iter_num: int = 0) -> ActionDecision:
    return ActionDecision(
        experiment_id=f"exp_{iter_num}",
        action_type=action_type,
        parameters={"model_name": model_name},
        preprocessing=PreprocessingConfig(
            missing_value_strategy="mean",
            scaling="none",
            encoding="none",
            imbalance_strategy="none",
            feature_selection="none",
        ),
        reason="test",
        expected_gain=0.01,
        expected_cost=1.0,
        confidence=0.8,
    )


class TestBudgetEnforcement:
    """Tests A through J for Phase 2 Budget Enforcement."""

    def test_a_budget_10_ceiling_never_exceeded(self, tmp_path):
        """Test A: Budget = 10 never exceeds 10 model evaluations."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_a_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            # Never emit terminate, always propose another action
            return _make_stub_action(model_name="DecisionTreeClassifier", action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=10,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations <= 10, f"Expected <= 10 evaluations, got {orchestrator.actual_evaluations}"
            assert orchestrator.actual_evaluations == 10, f"Expected exactly 10 evaluations, got {orchestrator.actual_evaluations}"
            assert orchestrator.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_b_budget_20_ceiling_never_exceeded(self, tmp_path):
        """Test B: Budget = 20 never exceeds 20 model evaluations."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_b_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            # Alternate models without terminating
            m = "DecisionTreeClassifier" if call_count % 2 == 1 else "LogisticRegression"
            return _make_stub_action(model_name=m, action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=20,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations <= 20, f"Expected <= 20 evaluations, got {orchestrator.actual_evaluations}"
            assert orchestrator.actual_evaluations == 20, f"Expected exactly 20 evaluations, got {orchestrator.actual_evaluations}"
            assert orchestrator.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_c_early_termination_before_budget(self, tmp_path):
        """Test C: Early termination terminates before budget when terminate action occurs."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_c_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return _make_stub_action(action_type="terminate", iter_num=call_count)
            return _make_stub_action(model_name="DecisionTreeClassifier", action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=10,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 3, f"Expected 3 evaluations, got {orchestrator.actual_evaluations}"
            assert orchestrator.actual_evaluations < 10
            assert orchestrator.termination_reason == "agent_terminated"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_d_budget_exhaustion_terminates_at_exact_ceiling(self, tmp_path):
        """Test D: Budget exhaustion automatically terminates at exactly the configured ceiling."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_d_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            return _make_stub_action(model_name="LogisticRegression", action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=4,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 4
            assert orchestrator.decision_iterations == 4
            assert orchestrator.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_e_duplicate_configuration_accounting(self, tmp_path):
        """Test E: Duplicate configuration counts as a model evaluation towards the budget ceiling."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_e_{uuid.uuid4().hex[:6]}"

        call_count = 0

        # Create real DecisionEngine to verify repeated_configs tracking
        engine = DecisionEngine(
            run_id=run_id,
            evaluation_budget=3,
            allowed_models=["LogisticRegression"],
        )

        def wrapped_result(result):
            nonlocal call_count
            call_count += 1
            action = engine.receive_result(result)
            # Force repeated configuration request
            action.action_type = "modify_regularization"
            action.parameters = {"model_name": "LogisticRegression"}
            return action

        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=wrapped_result,
            run_id=run_id,
            evaluation_budget=3,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 3
            assert engine._repeated_configs >= 1, "Repeated configurations must be tracked in DecisionEngine"
            assert orchestrator.termination_reason == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_f_failed_evaluation_accounting_no_retry_loop(self, tmp_path):
        """Test F: Failed evaluation increments evaluation count and cannot cause infinite retry loop."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_f_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            # Verify result was marked as failed
            assert result.failed is True
            assert result.status == "failed"
            return _make_stub_action(model_name="LogisticRegression", action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=3,
            tune=False,
        )

        # Mock run_ml_pipeline to simulate runtime failure
        with patch("stratml.orchestration.orchestrator.run_ml_pipeline", side_effect=RuntimeError("Simulated pipeline failure")):
            try:
                orchestrator.run(str(csv_path), target)
                assert orchestrator.actual_evaluations == 3, "Each failed evaluation must consume 1 evaluation from budget"
                assert orchestrator.actual_fits == 3
                assert orchestrator.termination_reason == "budget_exhausted"
            finally:
                shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_g_tune_false_accounting(self, tmp_path):
        """Test G: When tune=False, actual_evaluations == actual_fits == decision_iterations."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_g_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            return _make_stub_action(model_name="DecisionTreeClassifier", action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=3,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)
            assert orchestrator.actual_evaluations == 3
            assert orchestrator.actual_fits == 3
            assert orchestrator.decision_iterations == 3
            assert orchestrator.actual_evaluations == orchestrator.actual_fits == orchestrator.decision_iterations
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_h_tune_true_evaluations_vs_fits(self, tmp_path):
        """Test H: When tune=True, model fits exceed model evaluations due to CV folds."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_h_{uuid.uuid4().hex[:6]}"

        def stub_profile(profile):
            return _make_stub_action(model_name="RandomForestClassifier", iter_num=0)

        def stub_result(result):
            return _make_stub_action(action_type="terminate", iter_num=1)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=10,
            tune=True,
        )

        # Mock RandomizedSearchCV to verify fits vs evaluations calculation
        with patch("sklearn.model_selection.RandomizedSearchCV") as mock_cv:
            from sklearn.dummy import DummyClassifier
            underlying = DummyClassifier(strategy="most_frequent")
            mock_inst = MagicMock()

            def fake_fit_h(X, y):
                underlying.fit(X, y)
                return mock_inst

            mock_inst.fit.side_effect = fake_fit_h
            mock_inst.best_estimator_ = underlying
            mock_inst.cv_results_ = {"params": [{} for _ in range(5)]}
            mock_inst.n_splits_ = 3
            mock_inst.refit = True
            mock_cv.return_value = mock_inst

            try:
                orchestrator.run(str(csv_path), target)
                # 5 candidates * 3 folds + 1 refit = 16 fits; 5 evaluations
                assert orchestrator.actual_evaluations == 5
                assert orchestrator.actual_fits == 16
                assert orchestrator.actual_fits > orchestrator.actual_evaluations
            finally:
                shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_i_decision_iterations_differ_from_evaluation_count(self, tmp_path):
        """Test I: Decision iterations differ from evaluation count when multi-candidate tuning runs."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_i_{uuid.uuid4().hex[:6]}"

        def stub_profile(profile):
            return _make_stub_action(model_name="RandomForestClassifier", iter_num=0)

        def stub_result(result):
            return _make_stub_action(action_type="terminate", iter_num=1)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=10,
            tune=True,
        )

        with patch("sklearn.model_selection.RandomizedSearchCV") as mock_cv:
            from sklearn.dummy import DummyClassifier
            underlying = DummyClassifier(strategy="most_frequent")
            mock_inst = MagicMock()

            def fake_fit_i(X, y):
                underlying.fit(X, y)
                return mock_inst

            mock_inst.fit.side_effect = fake_fit_i
            mock_inst.best_estimator_ = underlying
            mock_inst.cv_results_ = {"params": [{} for _ in range(8)]}
            mock_inst.n_splits_ = 3
            mock_inst.refit = True
            mock_cv.return_value = mock_inst

            try:
                orchestrator.run(str(csv_path), target)
                assert orchestrator.decision_iterations == 1
                assert orchestrator.actual_evaluations == 8
                assert orchestrator.decision_iterations != orchestrator.actual_evaluations
            finally:
                shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_j_manifest_budget_records(self, tmp_path):
        """Test J: Manifest and budget accounting record configured budget, actual evaluations, fits, and termination reason."""
        csv_path, target = _make_dummy_dataset(tmp_path)
        run_id = f"test_j_{uuid.uuid4().hex[:6]}"

        call_count = 0

        def stub_profile(profile):
            return _make_stub_action(model_name="LogisticRegression", iter_num=0)

        def stub_result(result):
            nonlocal call_count
            call_count += 1
            return _make_stub_action(model_name="LogisticRegression", action_type="switch_model", iter_num=call_count)

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_profile,
            send_result=stub_result,
            run_id=run_id,
            evaluation_budget=5,
            tune=False,
        )

        try:
            orchestrator.run(str(csv_path), target)

            artifacts_dir = Path("outputs") / run_id / "artifacts"
            manifest_file = Path("outputs") / run_id / "manifest.json"
            budget_file = artifacts_dir / "budget_accounting.json"

            assert manifest_file.exists(), "manifest.json must exist"
            assert budget_file.exists(), "budget_accounting.json must exist"

            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            budget = json.loads(budget_file.read_text(encoding="utf-8"))

            # 1. Configured budget
            assert manifest["evaluation_budget"] == 5
            assert manifest["configured_budget"] == 5
            assert budget["configured_budget"]["evaluation_budget"] == 5

            # 2. Actual evaluations and fits
            assert manifest["actual_evaluations"] == 5
            assert manifest["actual_fits"] == 5
            assert budget["actual_consumption"]["model_evaluations"] == 5
            assert budget["actual_consumption"]["actual_evaluations"] == 5
            assert budget["actual_consumption"]["model_fits"] == 5

            # 3. Termination reason
            assert manifest["termination_reason"] == "budget_exhausted"
            assert budget["termination_reason"] == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)


class TestARFFDatasetSmokeTests:
    """Section 11: Dataset Smoke Tests on acquired ARFF datasets with verified budget accounting."""

    def test_binary_smoke_australian_budget(self):
        """Smoke test binary classification (australian.arff) with budget=2."""
        arff_path = Path("data/research/classification/australian.arff")
        if not arff_path.exists():
            pytest.skip("australian.arff not found in data/research/classification/")

        run_id = f"smoke_australian_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(
            run_id=run_id,
            evaluation_budget=2,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=2,
            tune=False,
        )

        try:
            orchestrator.run(str(arff_path), "A15")
            assert orchestrator.actual_evaluations == 2
            assert orchestrator.actual_fits == 2
            assert orchestrator.termination_reason == "budget_exhausted"

            manifest_file = Path("outputs") / run_id / "manifest.json"
            assert manifest_file.exists()
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert data["evaluation_budget"] == 2
            assert data["actual_evaluations"] == 2
            assert data["termination_reason"] == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_multiclass_smoke_vehicle_budget(self):
        """Smoke test multiclass classification (vehicle.arff) with budget=2."""
        arff_path = Path("data/research/classification/vehicle.arff")
        if not arff_path.exists():
            pytest.skip("vehicle.arff not found in data/research/classification/")

        run_id = f"smoke_vehicle_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(
            run_id=run_id,
            evaluation_budget=2,
            allowed_models=["LogisticRegression", "DecisionTreeClassifier"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=2,
            tune=False,
        )

        try:
            orchestrator.run(str(arff_path), "Class")
            assert orchestrator.actual_evaluations == 2
            assert orchestrator.actual_fits == 2
            assert orchestrator.termination_reason == "budget_exhausted"

            manifest_file = Path("outputs") / run_id / "manifest.json"
            assert manifest_file.exists()
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert data["evaluation_budget"] == 2
            assert data["actual_evaluations"] == 2
            assert data["termination_reason"] == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_regression_smoke_boston_budget(self):
        """Smoke test regression (boston.arff) with budget=2."""
        arff_path = Path("data/research/regression/boston.arff")
        if not arff_path.exists():
            pytest.skip("boston.arff not found in data/research/regression/")

        run_id = f"smoke_boston_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(
            run_id=run_id,
            evaluation_budget=2,
            allowed_models=["Ridge", "DecisionTreeRegressor"],
        )
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            evaluation_budget=2,
            tune=False,
        )

        try:
            orchestrator.run(str(arff_path), "MEDV")
            assert orchestrator.actual_evaluations == 2
            assert orchestrator.actual_fits == 2
            assert orchestrator.termination_reason == "budget_exhausted"

            manifest_file = Path("outputs") / run_id / "manifest.json"
            assert manifest_file.exists()
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert data["evaluation_budget"] == 2
            assert data["actual_evaluations"] == 2
            assert data["termination_reason"] == "budget_exhausted"
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)
