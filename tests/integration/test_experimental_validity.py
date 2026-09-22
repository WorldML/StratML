"""
test_experimental_validity.py
------------------------------
Paper experimental-validity suite:
1. Counterfactual impact determinism
2. Canonical action vocabulary and non-empty execution
3. Authoritative hyperparameter mutations (before != after)
4. Classification vs Regression task & metric adaptation
5. Holdout boundary enforcement (test split untouched in loop)
6. Computational budget fidelity (no hidden CV when tune=False)
"""

import numpy as np
import pandas as pd
import pytest

from stratml.core.schemas import (
    ActionDecision,
    CandidateAction,
    DecisionReason,
    PreprocessingConfig,
)
from stratml.decision.agents.evaluator_agent import _rule_audit
from stratml.decision.engine import DecisionEngine
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.schemas import DataProfile, DataSplit, FeatureInfo
from tests.unit.test_decision_team import _make_state, _make_result


@pytest.fixture(autouse=True)
def disable_external_apis(monkeypatch):
    """Ensure experimental validity tests run offline without hitting external APIs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _make_dummy_split():
    np.random.seed(42)
    X_tr = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "cat": ["x", "y", "x", "y"]})
    y_tr = pd.Series([0, 1, 0, 1], name="target")
    X_va = pd.DataFrame({"a": [2.0, np.nan, 3.0, 1.0], "cat": ["y", "x", "y", "x"]})
    y_va = pd.Series([1, 0, 1, 0], name="target")
    X_te = pd.DataFrame({"a": [999.0, 999.0, 999.0, 999.0], "cat": ["x", "x", "x", "x"]})
    y_te = pd.Series([0, 0, 0, 0], name="target")
    return DataSplit(X_train=X_tr, X_val=X_va, X_test=X_te, y_train=y_tr, y_val=y_va, y_test=y_te)


def _make_dummy_profile(problem_type="classification"):
    return DataProfile(
        dataset_name="test_data",
        dataset_type="tabular",
        rows=100,
        columns=3,
        target_column="target",
        problem_type=problem_type,
        numerical_columns=["a"],
        categorical_columns=["cat"],
        missing_value_ratio=0.1,
        feature_summary=[
            FeatureInfo(name="a", dtype="float64", unique_values=4, missing_percentage=0.1, distribution="uniform"),
            FeatureInfo(name="cat", dtype="object", unique_values=2, missing_percentage=0.0, distribution="uniform"),
        ],
        recommended_metrics=["accuracy"] if problem_type == "classification" else ["r2"],
    )


class TestP0_1_CounterfactualMath:
    @pytest.mark.parametrize("previous,current,expected,expected_cf", [
        (0.80, 0.82, 0.01, 0.01),    # (0.82 - 0.80) - 0.01 = +0.01
        (0.80, 0.81, 0.01, 0.00),    # (0.81 - 0.80) - 0.01 =  0.00
        (0.80, 0.78, 0.01, -0.03),   # (0.78 - 0.80) - 0.01 = -0.03
    ])
    def test_exact_values(self, previous, current, expected, expected_cf):
        decision = ActionDecision(
            experiment_id="exp1",
            iteration=1,
            action_type="switch_model",
            parameters={"model_name": "LogisticRegression"},
            preprocessing=PreprocessingConfig(
                missing_value_strategy="mean", scaling="standard",
                encoding="onehot", imbalance_strategy="none", feature_selection="none",
            ),
            expected_gain=expected,
            reason=DecisionReason(
                trigger="underfitting",
                evidence={"best_score": previous, "primary_metric": "accuracy"},
                source="rule",
            ),
        )
        result = _make_result(accuracy=current)
        state = _make_state(primary=current, train_loss=0.20, val_loss=0.22)
        rec = _rule_audit(decision, result, state)
        assert rec.counterfactual_impact == pytest.approx(expected_cf, abs=1e-4)


class TestP0_2_And_P0_6_ActionExecutionFidelity:
    def test_add_preprocessing_modifies_config(self):
        prep = PreprocessingConfig(
            missing_value_strategy="mean", scaling="none",
            encoding="none", imbalance_strategy="none", feature_selection="none",
        )
        decision = ActionDecision(
            experiment_id="exp1",
            action_type="add_preprocessing",
            parameters={"model_name": "LogisticRegression", "strategy": "oversample"},
            preprocessing=prep,
            reason="overfitting",
        )
        config = build_experiment_config(decision)
        assert config.preprocessing.imbalance_strategy == "oversample"

    def test_classical_models_reject_dl_actions(self):
        for dl_action in ["change_optimizer", "unfreeze_backbone", "switch_architecture"]:
            decision = ActionDecision(
                experiment_id="exp1",
                action_type=dl_action,
                parameters={"model_name": "RandomForestClassifier"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean", scaling="none",
                    encoding="none", imbalance_strategy="none", feature_selection="none",
                ),
                reason="divergence",
            )
            with pytest.raises(ValueError, match="only supported for deep learning"):
                build_experiment_config(decision)


class TestP0_3_HyperparameterMutations:
    @pytest.mark.parametrize("model_name,param", [
        ("RandomForestClassifier", "n_estimators"),
        ("GradientBoostingClassifier", "n_estimators"),
        ("DecisionTreeClassifier", "max_depth"),
        ("LogisticRegression", "C"),
        ("SVC", "C"),
        ("KNeighborsClassifier", "n_neighbors"),
    ])
    def test_all_classical_models_mutate_deterministically(self, model_name, param):
        prep = PreprocessingConfig(
            missing_value_strategy="mean", scaling="none",
            encoding="none", imbalance_strategy="none", feature_selection="none",
        )
        inc_decision = ActionDecision(
            experiment_id="exp1",
            action_type="increase_model_capacity",
            parameters={"model_name": model_name, "scale": 1.5},
            preprocessing=prep,
            reason="underfitting",
        )
        dec_decision = ActionDecision(
            experiment_id="exp1",
            action_type="decrease_model_capacity",
            parameters={"model_name": model_name, "scale": 0.75},
            preprocessing=prep,
            reason="overfitting",
        )
        inc_cfg = build_experiment_config(inc_decision)
        dec_cfg = build_experiment_config(dec_decision)

        assert inc_cfg.hyperparameters[param] != dec_cfg.hyperparameters[param]


class TestP0_4_TaskAndMetricAdaptation:
    def test_regression_profile_adapts_engine_to_r2(self):
        engine = DecisionEngine()
        reg_profile = _make_dummy_profile(problem_type="regression")
        decision = engine.receive_profile(reg_profile)

        assert engine.primary_metric == "r2"
        # First decision should select a regression model, not a classifier
        assert "Regressor" in decision.parameters["model_name"] or decision.parameters["model_name"] in ["Ridge", "Lasso", "ElasticNet", "LinearRegression"]

    def test_classification_profile_keeps_accuracy(self):
        engine = DecisionEngine()
        clf_profile = _make_dummy_profile(problem_type="classification")
        decision = engine.receive_profile(clf_profile)

        assert engine.primary_metric == "accuracy"
        assert "Classifier" in decision.parameters["model_name"] or decision.parameters["model_name"] in ["LogisticRegression", "GaussianNB"]


class TestP0_5_HoldoutBoundary:
    def test_experimentation_loop_does_not_touch_test_split(self):
        split = _make_dummy_split()
        original_test_a = split.X_test["a"].copy()
        profile = _make_dummy_profile()
        prep = PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard",
            encoding="onehot", imbalance_strategy="none", feature_selection="none",
        )

        # Loop call: transform_test=False
        clean_split, _ = apply_preprocessing(split, prep, profile, transform_test=False)

        # Assert X_test was NOT transformed or modified
        pd.testing.assert_series_equal(clean_split.X_test["a"], original_test_a)

        # Final evaluation call: transform_test=True
        final_split, _ = apply_preprocessing(split, prep, profile, transform_test=True)

        # Assert X_test WAS transformed (e.g. scaled)
        assert not clean_split.X_test.equals(final_split.X_test)


def _make_clean_numeric_split():
    np.random.seed(42)
    X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    y_tr = pd.Series([0, 1, 0, 1], name="target")
    X_va = pd.DataFrame({"a": [2.0, 1.0, 3.0, 2.0], "b": [0.2, 0.1, 0.4, 0.3]})
    y_va = pd.Series([1, 0, 1, 0], name="target")
    X_te = pd.DataFrame({"a": [1.5, 2.5, 3.5, 4.5], "b": [0.15, 0.25, 0.35, 0.45]})
    y_te = pd.Series([0, 1, 0, 1], name="target")
    return DataSplit(X_train=X_tr, X_val=X_va, X_test=X_te, y_train=y_tr, y_val=y_va, y_test=y_te)


class FakeModel1:
    def __init__(self):
        self.name = "Model1"

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class FakeModel2:
    def __init__(self):
        self.name = "Model2"

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class TestP0_6_FinalArtifactSelection:
    def test_best_validation_model_and_preprocessing_preserved_and_used_for_test(self):
        import json
        import shutil
        import uuid
        from pathlib import Path
        from unittest.mock import patch

        import joblib

        from stratml.execution.pipelines.ml_pipeline import MLPipelineResult
        from stratml.orchestration.orchestrator import ExecutionOrchestrator

        run_id = f"test_art_{uuid.uuid4().hex[:8]}"
        call_count = 0

        def fake_run_ml_pipeline(config, clean_split):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Iteration 1: High validation accuracy (1.0)
                m = FakeModel1()
                return MLPipelineResult(
                    model=m,
                    y_val_pred=clean_split.y_val.values,
                    train_curve=[0.1],
                    val_curve=[0.1],
                    runtime=0.05,
                )
            else:
                # Iteration 2: Low validation accuracy (0.0)
                m = FakeModel2()
                bad_pred = np.full(len(clean_split.y_val), 999, dtype=int)
                return MLPipelineResult(
                    model=m,
                    y_val_pred=bad_pred,
                    train_curve=[0.5],
                    val_curve=[0.6],
                    runtime=0.05,
                )

        def stub_send_profile(profile):
            return ActionDecision(
                experiment_id=f"{run_id}_iter1",
                iteration=0,
                action_type="switch_model",
                parameters={"model_name": "LogisticRegression"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="standard",
                    encoding="onehot",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                reason="test",
                expected_gain=0.1,
                expected_cost=1.0,
                confidence=1.0,
            )

        def stub_send_result(result):
            if result.iteration == 1:
                return ActionDecision(
                    experiment_id=f"{run_id}_iter2",
                    iteration=1,
                    action_type="switch_model",
                    parameters={"model_name": "RandomForestClassifier"},
                    preprocessing=PreprocessingConfig(
                        missing_value_strategy="mean",
                        scaling="minmax",
                        encoding="onehot",
                        imbalance_strategy="none",
                        feature_selection="none",
                    ),
                    reason="test",
                    expected_gain=0.05,
                    expected_cost=1.0,
                    confidence=0.8,
                )
            else:
                return ActionDecision(
                    experiment_id=f"{run_id}_iter3",
                    iteration=2,
                    action_type="terminate",
                    parameters={},
                    preprocessing=PreprocessingConfig(
                        missing_value_strategy="mean",
                        scaling="minmax",
                        encoding="onehot",
                        imbalance_strategy="none",
                        feature_selection="none",
                    ),
                    reason="test",
                    expected_gain=0.0,
                    expected_cost=0.0,
                    confidence=1.0,
                )

        orchestrator = ExecutionOrchestrator(
            send_profile=stub_send_profile,
            send_result=stub_send_result,
            run_id=run_id,
        )

        try:
            with patch(
                "stratml.orchestration.orchestrator.run_ml_pipeline",
                side_effect=fake_run_ml_pipeline,
            ):
                orchestrator.run("data/raw/iris.csv", "species")

            artifacts_dir = Path("outputs") / run_id / "artifacts"
            root_model_path = artifacts_dir / "model.pkl"
            root_config_path = artifacts_dir / "config.json"
            test_metrics_path = artifacts_dir / "test_metrics.json"

            assert root_model_path.exists(), "Root model.pkl must exist"
            assert root_config_path.exists(), "Root config.json must exist"
            assert test_metrics_path.exists(), "Test metrics must be evaluated and saved"

            # 1. Root model must be Model 1 (from best validation iteration 1, not worse iteration 2)
            saved_model = joblib.load(root_model_path)
            assert isinstance(saved_model, FakeModel1)
            assert saved_model.name == "Model1"

            # 2. Root config must match iteration 1's preprocessing
            config_data = json.loads(root_config_path.read_text())
            assert config_data["model_name"] == "LogisticRegression"
            assert config_data["preprocessing"]["scaling"] == "standard"
            assert config_data["preprocessing"]["scaling"] != "minmax"

            # 3. Test metrics must be present and correspond to evaluating Model 1
            test_metrics_data = json.loads(test_metrics_path.read_text())
            assert test_metrics_data["accuracy"] is not None
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)


class TestP0_7_ComputationalBudget:
    def test_tune_false_never_runs_randomized_search_cv(self):
        from unittest.mock import patch

        from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline

        split = _make_clean_numeric_split()
        prep = PreprocessingConfig(
            missing_value_strategy="mean",
            scaling="none",
            encoding="none",
            imbalance_strategy="none",
            feature_selection="none",
        )
        decision = ActionDecision(
            experiment_id="exp_tune_false",
            action_type="switch_model",
            parameters={"model_name": "RandomForestClassifier"},
            preprocessing=prep,
            reason="test",
        )
        config = build_experiment_config(decision, tune=False)
        assert config.tune is False

        with patch("sklearn.model_selection.RandomizedSearchCV") as mock_cv:
            res = run_ml_pipeline(config, split)
            assert mock_cv.called is False
            assert res.model is not None

    def test_tune_true_uses_randomized_search_cv(self):
        from unittest.mock import MagicMock, patch

        from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline

        split = _make_clean_numeric_split()
        prep = PreprocessingConfig(
            missing_value_strategy="mean",
            scaling="none",
            encoding="none",
            imbalance_strategy="none",
            feature_selection="none",
        )
        decision = ActionDecision(
            experiment_id="exp_tune_true",
            action_type="switch_model",
            parameters={"model_name": "RandomForestClassifier"},
            preprocessing=prep,
            reason="test",
        )
        config = build_experiment_config(decision, tune=True)
        assert config.tune is True

        with patch("sklearn.model_selection.RandomizedSearchCV") as mock_cv:
            mock_inst = MagicMock()
            mock_cv.return_value = mock_inst
            mock_inst.best_estimator_ = MagicMock()
            run_ml_pipeline(config, split)
            assert mock_cv.called is True

    @pytest.mark.parametrize("budget", [1, 2, 4])
    def test_iteration_budget_maps_to_exact_executions(self, budget):
        engine = DecisionEngine(max_iterations=budget)
        profile = _make_dummy_profile()

        # Step 0: bootstrap
        action = engine.receive_profile(profile)
        assert action.action_type != "terminate"

        # Simulate execution results for each iteration
        for i in range(1, budget):
            result = _make_result(accuracy=0.75 + 0.01 * i)
            result.iteration = i
            action = engine.receive_result(result)
            assert action.action_type != "terminate", f"Prematurely terminated at iteration {i} of {budget}"

        # On the N-th iteration, budget is exhausted -> MUST terminate
        result_final = _make_result(accuracy=0.85)
        result_final.iteration = budget
        action_final = engine.receive_result(result_final)
        assert (
            action_final.action_type == "terminate"
        ), f"Expected terminate at budget {budget}, got {action_final.action_type}"

    def test_orchestrator_engine_end_to_end_budget_fidelity(self):
        import shutil
        import uuid
        from pathlib import Path

        from stratml.orchestration.orchestrator import ExecutionOrchestrator

        run_id = f"test_budget_{uuid.uuid4().hex[:8]}"
        engine = DecisionEngine(max_iterations=2)
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            tune=False,
        )
        try:
            orchestrator.run("data/raw/iris.csv", "species")
            # Verify iter_1 and iter_2 directories exist, but NOT iter_3
            iter1_dir = Path("outputs") / run_id / "artifacts" / "iter_1"
            iter2_dir = Path("outputs") / run_id / "artifacts" / "iter_2"
            iter3_dir = Path("outputs") / run_id / "artifacts" / "iter_3"

            assert iter1_dir.exists(), "Iteration 1 artifacts must exist"
            assert iter2_dir.exists(), "Iteration 2 artifacts must exist"
            assert not iter3_dir.exists(), "Iteration 3 must NOT exist when max_iterations=2"

            # Root artifacts and test metrics must exist
            assert (Path("outputs") / run_id / "artifacts" / "model.pkl").exists()
            assert (Path("outputs") / run_id / "artifacts" / "test_metrics.json").exists()
        finally:
            shutil.rmtree(Path("outputs") / run_id, ignore_errors=True)


class TestP1_EvaluatorCoordinatorBehavior:
    """P1-7: Evaluator -> Coordinator behavioral integration.

    Verifies that post-hoc audit records produced by the evaluator agent
    update the coordinator's EMA agent weights and actively reorder action rankings.
    """

    def test_default_weights_when_under_threshold(self, tmp_path):
        from stratml.decision.agents.coordinator_agent import _load_agent_weights
        import json

        # Non-existent or empty log file
        w_p, w_e, w_s = _load_agent_weights(log_paths=[tmp_path / "non_existent.jsonl"])
        assert (w_p, w_e, w_s) == (0.50, 0.25, 0.25)

        # Fewer than 5 records -> still default
        log_file = tmp_path / "few_records.jsonl"
        with open(log_file, "w") as f:
            for i in range(4):
                f.write(
                    json.dumps({
                        "counterfactual_impact": 0.05,
                        "decision_validity": 0.9,
                        "quality_risk": 0.1,
                    })
                    + "\n"
                )
        w_p, w_e, w_s = _load_agent_weights(log_paths=[log_file])
        assert (w_p, w_e, w_s) == (0.50, 0.25, 0.25)

    def test_scenario_a_performance_right_shifts_weights_and_reorders_actions(self, tmp_path):
        """Scenario A: Performance agent right, Stability agent wrong.
        Audit records reflect high validity & positive cf_impact, but high quality risk.
        -> w_p increases, w_s decreases -> high-performance action is ranked #1.
        """
        import json
        from stratml.decision.agents import coordinator_agent
        from stratml.decision.learning.uncertainty import UncertaintyEstimate

        log_file = tmp_path / "scenario_a.jsonl"
        records = [
            {"counterfactual_impact": 0.05, "decision_validity": 0.9, "quality_risk": 0.8}
            for _ in range(8)
        ]
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        w_p, w_e, w_s = coordinator_agent._load_agent_weights(log_paths=[log_file])
        assert w_p > w_s, f"Expected w_p > w_s, got w_p={w_p}, w_s={w_s}"
        assert w_p > 0.45

        state = _make_state(primary=0.80, train_loss=0.20, val_loss=0.22)
        cand_perf = UncertaintyEstimate(
            action_type="increase_model_capacity",
            parameters={"n_estimators": 200},
            predicted_gain=0.05,
            predicted_cost=2.0,
            confidence=0.8,
            variance=0.01,
        )
        cand_stab = UncertaintyEstimate(
            action_type="modify_regularization",
            parameters={"min_samples_split": 5},
            predicted_gain=0.01,
            predicted_cost=1.0,
            confidence=0.8,
            variance=0.01,
        )

        perf_scores = {"increase_model_capacity": 0.9, "modify_regularization": 0.1}
        eff_scores = {"increase_model_capacity": 0.5, "modify_regularization": 0.5}
        stab_scores = {"increase_model_capacity": 0.1, "modify_regularization": 0.9}

        ranked = coordinator_agent.rank(
            state,
            [cand_perf, cand_stab],
            perf_scores,
            eff_scores,
            stab_scores,
            log_paths=[log_file],
        )

        assert len(ranked) == 2
        assert ranked[0].action_type == "increase_model_capacity"
        assert ranked[0].final_score > ranked[1].final_score

    def test_scenario_b_stability_right_shifts_weights_and_reorders_actions(self, tmp_path):
        """Scenario B: Stability agent right, Performance agent wrong.
        Audit records reflect low validity, negative cf_impact, but low quality risk.
        -> w_s increases, w_p decreases -> high-stability action is ranked #1.
        """
        import json
        from stratml.decision.agents import coordinator_agent
        from stratml.decision.learning.uncertainty import UncertaintyEstimate

        log_file = tmp_path / "scenario_b.jsonl"
        records = [
            {"counterfactual_impact": -0.08, "decision_validity": 0.2, "quality_risk": 0.1}
            for _ in range(8)
        ]
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        w_p, w_e, w_s = coordinator_agent._load_agent_weights(log_paths=[log_file])
        assert w_s > w_p, f"Expected w_s > w_p, got w_s={w_s}, w_p={w_p}"
        assert w_s > 0.45

        state = _make_state(primary=0.80, train_loss=0.20, val_loss=0.22)
        cand_perf = UncertaintyEstimate(
            action_type="increase_model_capacity",
            parameters={"n_estimators": 200},
            predicted_gain=0.05,
            predicted_cost=2.0,
            confidence=0.8,
            variance=0.01,
        )
        cand_stab = UncertaintyEstimate(
            action_type="modify_regularization",
            parameters={"min_samples_split": 5},
            predicted_gain=0.01,
            predicted_cost=1.0,
            confidence=0.8,
            variance=0.01,
        )

        perf_scores = {"increase_model_capacity": 0.9, "modify_regularization": 0.1}
        eff_scores = {"increase_model_capacity": 0.5, "modify_regularization": 0.5}
        stab_scores = {"increase_model_capacity": 0.1, "modify_regularization": 0.9}

        ranked = coordinator_agent.rank(
            state,
            [cand_perf, cand_stab],
            perf_scores,
            eff_scores,
            stab_scores,
            log_paths=[log_file],
        )

        assert len(ranked) == 2
        assert ranked[0].action_type == "modify_regularization"
        assert ranked[0].final_score > ranked[1].final_score

    def test_evaluator_audit_writes_valid_audit_record(self, tmp_path):
        """Direct test of evaluator_agent.audit writing valid JSON Lines."""
        import json
        from stratml.decision.agents import evaluator_agent

        log_file = tmp_path / "audit_test.jsonl"
        orig_log = evaluator_agent._EVAL_LOG
        evaluator_agent._EVAL_LOG = log_file

        try:
            decision = ActionDecision(
                experiment_id="exp_test_audit",
                iteration=1,
                action_type="switch_model",
                parameters={"model_name": "RandomForestClassifier"},
                preprocessing=PreprocessingConfig(
                    missing_value_strategy="mean",
                    scaling="none",
                    encoding="none",
                    imbalance_strategy="none",
                    feature_selection="none",
                ),
                expected_gain=0.02,
                reason=DecisionReason(
                    trigger="underfitting",
                    evidence={"best_score": 0.70, "primary_metric": "accuracy"},
                    source="rule",
                ),
            )
            result = _make_result(accuracy=0.80)
            state = _make_state(primary=0.80, train_loss=0.15, val_loss=0.18)

            rec = evaluator_agent.audit(decision, result, state)
            assert rec.action_type == "switch_model"
            assert log_file.exists()

            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["experiment_id"] == "exp_test_audit"
            assert entry["action_type"] == "switch_model"
            assert "decision_validity" in entry
            assert "counterfactual_impact" in entry
            assert "quality_risk" in entry
        finally:
            evaluator_agent._EVAL_LOG = orig_log

    def test_end_to_end_evaluator_coordinator_feedback_via_engine(self):
        """Verify DecisionEngine runs evaluator_agent.audit on receive_result and appends to log."""
        from pathlib import Path
        import shutil
        import uuid

        run_id = f"test_eval_coord_{uuid.uuid4().hex[:8]}"
        out_dir = Path("outputs") / run_id
        try:
            engine = DecisionEngine(run_id=run_id, max_iterations=3)
            profile = _make_dummy_profile()

            # Step 0: bootstrap
            action = engine.receive_profile(profile)
            assert action.action_type != "terminate"

            # Step 1: receive result -> triggers evaluator_agent.audit
            result1 = _make_result(accuracy=0.75)
            result1.iteration = 1
            action2 = engine.receive_result(result1)
            assert action2 is not None

            eval_log = out_dir / "decision_logs" / "evaluation_log.jsonl"
            assert eval_log.exists(), "Engine must generate evaluation_log.jsonl on receive_result"

            # Step 2: receive result -> appends second audit record
            result2 = _make_result(accuracy=0.80)
            result2.iteration = 2
            action3 = engine.receive_result(result2)
            assert action3 is not None

            lines = eval_log.read_text().strip().split("\n")
            assert len(lines) == 2, f"Expected 2 audit records, got {len(lines)}"
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

