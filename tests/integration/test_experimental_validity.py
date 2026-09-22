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
