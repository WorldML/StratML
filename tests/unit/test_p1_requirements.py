"""
test_p1_requirements.py
-----------------------
Focused test suite for P1 requirements (11-20):
  - P1-1 (Requirement #12): Frozen hyperparameter mutation space specification,
    isolation from RandomizedSearchCV tuning space, domain enforcement, and
    unsupported model/action rejection.
  - P1-2 (Requirement #20): Computational-budget definition hardening,
    actual model fit and candidate evaluation counting, paper compliance boundary,
    soft timeout semantics, and budget accounting artifacts/reports.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stratml.cli.config import DEFAULT_CONFIG
from stratml.execution.config import ml_mutations
from stratml.execution.config.ml_mutations import (
    PAPER_MUTATION_SPACE,
    get_model_mutation_spec,
    get_paper_mutation_space,
    is_mutation_supported,
    mutate_regularization,
    increase_capacity,
    decrease_capacity,
)
from stratml.execution.pipelines.ml_pipeline import (
    PAPER_CLASSICAL_MODELS,
    PAPER_CLASSIFICATION_MODELS,
    PAPER_REGRESSION_MODELS,
    _PARAM_GRIDS,
    run_ml_pipeline,
    MLPipelineResult,
)
from stratml.execution.schemas import (
    ActionDecision,
    DataSplit,
    ExperimentConfig,
    PreprocessingConfig,
    SplitConfig,
)
from stratml.orchestration.orchestrator import ExecutionOrchestrator


# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

def _make_dummy_split() -> DataSplit:
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.choice([0, 1], size=60), name="target")
    return DataSplit(
        X_train=X.iloc[:40],
        X_val=X.iloc[40:50],
        X_test=X.iloc[50:],
        y_train=y.iloc[:40],
        y_val=y.iloc[40:50],
        y_test=y.iloc[50:],
    )


# ===========================================================================
# P1-1: Requirement #12 — Freeze Exact Hyperparameter Mutation Space
# ===========================================================================

class TestP1_1_FrozenHyperparameterMutationSpace:
    def test_mutation_space_covers_all_sixteen_paper_classical_models(self):
        """PAPER_MUTATION_SPACE must explicitly map all 16 paper models."""
        assert len(PAPER_CLASSICAL_MODELS) == 16
        for model_name in PAPER_CLASSICAL_MODELS:
            assert model_name in PAPER_MUTATION_SPACE, (
                f"Paper model '{model_name}' missing from PAPER_MUTATION_SPACE"
            )

    def test_mutation_space_machine_readable_structure(self):
        """Every entry must specify param_type, domain (min/max), and supported_actions."""
        space = get_paper_mutation_space()
        for model_name, params in space.items():
            assert len(params) >= 1, f"{model_name} has no mutable parameters declared"
            for param_name, param_info in params.items():
                assert "param_type" in param_info, f"Missing param_type in {model_name}.{param_name}"
                assert param_info["param_type"] in ("int", "float")
                assert "domain" in param_info, f"Missing domain in {model_name}.{param_name}"
                assert "min" in param_info["domain"]
                assert "max" in param_info["domain"]
                assert "supported_actions" in param_info, (
                    f"Missing supported_actions in {model_name}.{param_name}"
                )
                actions = param_info["supported_actions"]
                assert any(
                    act in ("modify_regularization", "increase_model_capacity", "decrease_model_capacity")
                    for act in actions
                )

    def test_unsupported_models_rejected_in_mutate_regularization(self):
        """mutate_regularization must raise ValueError for unknown or unsupported models."""
        with pytest.raises(ValueError, match="does not support regularization mutations"):
            mutate_regularization("UnknownClassifier", {"C": 1.0}, "increase")

        with pytest.raises(ValueError, match="does not support regularization mutations"):
            mutate_regularization("LinearRegression", {}, "increase")

    def test_invalid_direction_rejected_in_mutate_regularization(self):
        """mutate_regularization must reject directions other than increase or decrease."""
        with pytest.raises(ValueError, match="Invalid direction"):
            mutate_regularization("LogisticRegression", {"C": 1.0}, "sideways")

    def test_unsupported_models_rejected_in_capacity_mutations(self):
        """increase_capacity and decrease_capacity must raise ValueError for unknown models."""
        with pytest.raises(ValueError, match="does not support capacity mutations"):
            increase_capacity("LinearRegression", {})

        with pytest.raises(ValueError, match="does not support capacity mutations"):
            decrease_capacity("LinearRegression", {})

        with pytest.raises(ValueError, match="does not support capacity mutations"):
            increase_capacity("CompletelyUnknownModel", {})

        with pytest.raises(ValueError, match="does not support capacity mutations"):
            decrease_capacity("CompletelyUnknownModel", {})

    @pytest.mark.parametrize("model_name", PAPER_CLASSICAL_MODELS)
    def test_all_paper_models_execute_valid_mutations_within_domain(self, model_name):
        """Every supported paper model must execute mutations producing values within declared domain."""
        spec = get_model_mutation_spec(model_name)

        # 1. Regularization mutation
        if is_mutation_supported(model_name, "modify_regularization"):
            reg_inc = mutate_regularization(model_name, {}, "increase")
            reg_dec = mutate_regularization(model_name, {}, "decrease")
            assert reg_inc != reg_dec, f"{model_name} regularization mutations produced identical hyperparameters"
            for p, p_spec in spec.items():
                if "modify_regularization" in p_spec["supported_actions"]:
                    min_val = p_spec["domain"]["min"]
                    if min_val is not None:
                        assert reg_inc[p] >= min_val
                        assert reg_dec[p] >= min_val

        # 2. Capacity mutations
        if is_mutation_supported(model_name, "increase_model_capacity"):
            cap_inc = increase_capacity(model_name, {}, scale=1.5)
            cap_dec = decrease_capacity(model_name, {}, scale=0.75)
            assert cap_inc != cap_dec, f"{model_name} capacity mutations produced identical hyperparameters"
            for p, p_spec in spec.items():
                if "increase_model_capacity" in p_spec["supported_actions"]:
                    min_val = p_spec["domain"]["min"]
                    if min_val is not None:
                        assert cap_inc[p] >= min_val
                        assert cap_dec[p] >= min_val

    def test_mutation_space_is_strictly_isolated_from_tuning_grids(self):
        """StratML mutation space != RandomizedSearchCV tuning grids."""
        # Tuning grids contain keys not present in classical mutation space
        assert "solver" in _PARAM_GRIDS["LogisticRegression"]
        assert "C" in get_model_mutation_spec("LogisticRegression")
        assert "solver" not in get_model_mutation_spec("LogisticRegression")

        assert "kernel" in _PARAM_GRIDS["SVC"]
        assert "kernel" not in get_model_mutation_spec("SVC")

        # GaussianNB has no tuning grid in _PARAM_GRIDS, but has mutation spec
        assert "GaussianNB" not in _PARAM_GRIDS
        assert "GaussianNB" in PAPER_MUTATION_SPACE
        assert "var_smoothing" in get_model_mutation_spec("GaussianNB")

    def test_mutation_space_getters_return_isolated_copies(self):
        """Mutating the dict returned by get_paper_mutation_space must not alter the master spec."""
        space_copy = get_paper_mutation_space()
        space_copy["RandomForestClassifier"]["max_depth"]["domain"]["min"] = -999
        fresh_space = get_paper_mutation_space()
        assert fresh_space["RandomForestClassifier"]["max_depth"]["domain"]["min"] == 1


# ===========================================================================
# P1-2: Requirement #20 — Harden Computational Budget Definition
# ===========================================================================

class TestP1_2_ComputationalBudgetHardening:
    def test_default_config_enforces_paper_budget(self):
        """Default CLI configuration must enforce max_iterations: 5 and tune: false."""
        exec_cfg = DEFAULT_CONFIG["execution"]
        assert exec_cfg["max_iterations"] == 5
        assert exec_cfg["tune"] is False
        assert exec_cfg["timeout_per_run"] == 300

    def test_ml_pipeline_result_records_exact_fit_and_eval_counts_tune_false(self):
        """When tune=False, MLPipelineResult must record fit_count=1 and eval_count=1."""
        split = _make_dummy_split()
        config = ExperimentConfig(
            experiment_id="test_exp_no_tune",
            model_name="RandomForestClassifier",
            model_type="ml",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            preprocessing=PreprocessingConfig(
                missing_value_strategy="mean",
                scaling="none",
                encoding="none",
                imbalance_strategy="none",
                feature_selection="none",
            ),
            tune=False,
            seed=42,
        )
        res = run_ml_pipeline(config, split)
        assert isinstance(res, MLPipelineResult)
        assert res.fit_count == 1
        assert res.eval_count == 1

    def test_ml_pipeline_result_records_exact_fit_and_eval_counts_tune_true(self):
        """When tune=True, MLPipelineResult must record actual CV fits and candidate evaluations."""
        split = _make_dummy_split()
        config = ExperimentConfig(
            experiment_id="test_exp_tuned",
            model_name="RandomForestClassifier",
            model_type="ml",
            hyperparameters={},
            preprocessing=PreprocessingConfig(
                missing_value_strategy="mean",
                scaling="none",
                encoding="none",
                imbalance_strategy="none",
                feature_selection="none",
            ),
            tune=True,
            seed=42,
        )
        # Mock RandomizedSearchCV to avoid heavy fitting while verifying count propagation
        with patch("sklearn.model_selection.RandomizedSearchCV") as mock_cv:
            mock_inst = MagicMock()
            mock_cv.return_value = mock_inst
            mock_inst.best_estimator_ = MagicMock()
            mock_inst.cv_results_ = {"params": [{} for _ in range(10)]}
            mock_inst.n_splits_ = 3
            mock_inst.refit = True

            res = run_ml_pipeline(config, split)
            # 10 candidates * 3 folds + 1 refit = 31 fits; 10 candidate evaluations
            assert res.fit_count == 31
            assert res.eval_count == 10

    def test_orchestrator_creates_budget_accounting_artifact_paper_standard(self, tmp_path):
        """Orchestrator must save budget_accounting.json identifying paper standard compliance."""
        run_id = f"test_budget_paper_{Path(tempfile.mkdtemp()).stem}"

        def stub_send_profile(profile):
            return ActionDecision(
                experiment_id=f"{run_id}_0",
                iteration=0,
                action_type="switch_model",
                parameters={"model_name": "LogisticRegression"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="bootstrap",
                expected_gain=0.1,
                expected_cost=1.0,
                confidence=1.0,
            )

        call_iter = 0

        def stub_send_result(result):
            nonlocal call_iter
            call_iter += 1
            if call_iter >= 2:
                return ActionDecision(
                    experiment_id=f"{run_id}_term",
                    iteration=call_iter,
                    action_type="terminate",
                    parameters={},
                    preprocessing=PreprocessingConfig(
                        missing_value_strategy="mean",
                        scaling="none",
                        encoding="none",
                        imbalance_strategy="none",
                        feature_selection="none",
                    ),
                    reason="budget_done",
                    expected_gain=0.0,
                    expected_cost=0.0,
                    confidence=1.0,
                )
            return ActionDecision(
                experiment_id=f"{run_id}_{call_iter}",
                iteration=call_iter,
                action_type="modify_regularization",
                parameters={"model_name": "LogisticRegression", "direction": "increase"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="regularize",
                expected_gain=0.05,
                expected_cost=1.0,
                confidence=0.8,
            )

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_send_profile,
            send_result=stub_send_result,
            run_id=run_id,
            max_iterations=5,
            tune=False,
            time_budget=300.0,
        )

        try:
            orchestrator.run("data/raw/iris.csv", "species")
            artifacts_dir = Path("outputs") / run_id / "artifacts"
            budget_file = artifacts_dir / "budget_accounting.json"
            assert budget_file.exists(), "budget_accounting.json was not created"

            data = json.loads(budget_file.read_text())

            # 1. Configured budget
            assert data["configured_budget"]["max_iterations"] == 5
            assert data["configured_budget"]["tune"] is False
            assert data["configured_budget"]["budget_type"] == "paper_standard"
            assert "soft_boundary" in data["configured_budget"]["timeout_semantics"]

            # 2. Actual consumption
            assert data["actual_consumption"]["decision_iterations"] == 2
            assert data["actual_consumption"]["model_fits"] == 2
            assert data["actual_consumption"]["model_evaluations"] == 2
            assert data["actual_consumption"]["runtime_seconds"] >= 0.0

            # 3. Paper compliance
            assert data["paper_compliance"]["is_paper_standard"] is True
            assert data["paper_compliance"]["boundary_status"] == "COMPLIANT_PAPER_STANDARD"
            assert data["paper_compliance"]["warning"] is None

        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_orchestrator_flags_tuning_in_budget_accounting_artifact(self):
        """When tune=True, budget_accounting.json must flag departure from paper budget."""
        run_id = f"test_budget_tuned_{Path(tempfile.mkdtemp()).stem}"

        def stub_send_profile(profile):
            return ActionDecision(
                experiment_id=f"{run_id}_0",
                iteration=0,
                action_type="switch_model",
                parameters={"model_name": "LogisticRegression"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="bootstrap",
                expected_gain=0.1,
                expected_cost=1.0,
                confidence=1.0,
            )

        def stub_send_result(result):
            return ActionDecision(
                experiment_id=f"{run_id}_term",
                iteration=1,
                action_type="terminate",
                parameters={},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="done",
                expected_gain=0.0,
                expected_cost=0.0,
                confidence=1.0,
            )

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_send_profile,
            send_result=stub_send_result,
            run_id=run_id,
            max_iterations=5,
            tune=True,
            time_budget=300.0,
        )

        try:
            orchestrator.run("data/raw/iris.csv", "species")
            artifacts_dir = Path("outputs") / run_id / "artifacts"
            budget_file = artifacts_dir / "budget_accounting.json"
            assert budget_file.exists()

            data = json.loads(budget_file.read_text())
            assert data["configured_budget"]["tune"] is True
            assert data["configured_budget"]["budget_type"] == "exploratory_tuned"
            assert data["paper_compliance"]["is_paper_standard"] is False
            assert data["paper_compliance"]["boundary_status"] == "EXPLORATORY_TUNING_ACTIVE"
            assert data["paper_compliance"]["warning"] is not None
            assert "--tune" in data["paper_compliance"]["warning"]

        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)

    def test_soft_timeout_completes_in_flight_iteration(self):
        """Soft timeout must finish current iteration cleanly before exiting."""
        run_id = f"test_soft_timeout_{Path(tempfile.mkdtemp()).stem}"

        def stub_send_profile(profile):
            return ActionDecision(
                experiment_id=f"{run_id}_0",
                iteration=0,
                action_type="switch_model",
                parameters={"model_name": "LogisticRegression"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="bootstrap",
                expected_gain=0.1,
                expected_cost=1.0,
                confidence=1.0,
            )

        # Never sends terminate: only the timeout will stop it
        def stub_send_result(result):
            return ActionDecision(
                experiment_id=f"{run_id}_{result.iteration}",
                iteration=result.iteration,
                action_type="modify_regularization",
                parameters={"model_name": "LogisticRegression", "direction": "increase"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="continue",
                expected_gain=0.01,
                expected_cost=1.0,
                confidence=0.5,
            )

        logs = []
        orchestrator = ExecutionOrchestrator(
            send_profile=stub_send_profile,
            send_result=stub_send_result,
            run_id=run_id,
            max_iterations=10,
            tune=False,
            time_budget=0.0001,  # immediate soft timeout after 1st iteration
            log=logs.append,
        )

        try:
            orchestrator.run("data/raw/iris.csv", "species")
            # Must have cleanly executed iteration 1 to completion
            assert orchestrator.decision_iterations >= 1
            assert orchestrator.actual_fits >= 1
            assert any("Soft timeout reached" in l for l in logs)

            budget_file = Path("outputs") / run_id / "artifacts" / "budget_accounting.json"
            assert budget_file.exists()
            data = json.loads(budget_file.read_text())
            assert data["actual_consumption"]["timeout_triggered"] is True
            assert "soft_boundary" in data["configured_budget"]["timeout_semantics"]
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)
