"""
test_config_builder.py
----------------------
Unit tests for execution/config/experiment_config_builder.py
"""

import inspect
import pytest

from stratml.decision.actions.action_generator import _DEFAULT_MODELS, _DEFAULT_REGRESSION_MODELS
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.pipelines.ml_pipeline import MODEL_REGISTRY
from stratml.execution.schemas import ActionDecision, PreprocessingConfig


def _action(action_type, params, prep=None):
    return ActionDecision(
        experiment_id="exp_test",
        action_type=action_type,
        parameters=params,
        preprocessing=prep or PreprocessingConfig(
            missing_value_strategy="mean", scaling="none",
            encoding="none", imbalance_strategy="none", feature_selection="none",
        ),
        reason="test",
        expected_gain=0.0,
        expected_cost=1.0,
        confidence=1.0,
    )


class TestActionTypeMapping:
    def test_switch_model_sets_model_name(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "Ridge"}))
        assert config.model_name == "Ridge"

    def test_switch_model_clears_hyperparams(self):
        # switch_model does a fresh start — extra params are dropped
        config = build_experiment_config(_action("switch_model", {"model_name": "Ridge", "alpha": 0.5}))
        assert config.hyperparameters == {}

    def test_increase_capacity(self):
        # scale=1.5 applied to default base of 100 -> 150
        config = build_experiment_config(_action("increase_model_capacity", {"model_name": "RandomForestClassifier", "scale": 1.5}))
        assert config.model_name == "RandomForestClassifier"
        assert config.hyperparameters.get("n_estimators") == 150

    def test_decrease_capacity(self):
        config = build_experiment_config(_action("decrease_model_capacity", {"model_name": "RandomForestClassifier", "n_estimators": 10}))
        assert config.model_name == "RandomForestClassifier"

    def test_modify_regularization(self):
        # direction="increase" reduces C by 0.1x from default 1.0 -> 0.1
        config = build_experiment_config(_action("modify_regularization", {"model_name": "LogisticRegression", "direction": "increase"}))
        assert config.hyperparameters.get("C") == pytest.approx(0.1, rel=1e-3)

    def test_change_optimizer(self):
        # lr_scale=0.1 applied to default lr=1e-3 -> 1e-4
        config = build_experiment_config(_action("change_optimizer", {"model_name": "MLP", "learning_rate_scale": 0.1}))
        assert config.hyperparameters.get("learning_rate") == pytest.approx(1e-4, rel=1e-3)

    def test_apply_preprocessing_keeps_model(self):
        config = build_experiment_config(_action("apply_preprocessing", {"model_name": "SVC"}))
        assert config.model_name == "SVC"

    def test_early_stop_sets_flag(self):
        config = build_experiment_config(_action("early_stop", {"model_name": "MLP", "early_stopping_patience": 7}))
        assert config.early_stopping is True
        assert config.early_stopping_patience == 7

    def test_terminate_does_not_raise(self):
        # terminate is handled gracefully — no ValueError
        config = build_experiment_config(_action("terminate", {}))
        assert config is not None


class TestModelTypeInference:
    def test_mlp_inferred_as_dl(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "MLP"}))
        assert config.model_type == "dl"

    def test_pytorch_mlp_inferred_as_dl(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "PyTorchMLP"}))
        assert config.model_type == "dl"

    def test_sklearn_model_inferred_as_ml(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "RandomForestClassifier"}))
        assert config.model_type == "ml"

    def test_ridge_inferred_as_ml(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "Ridge"}))
        assert config.model_type == "ml"


class TestPreprocessingPassthrough:
    def test_preprocessing_copied_to_config(self):
        prep = PreprocessingConfig(
            missing_value_strategy="median", scaling="minmax",
            encoding="onehot", imbalance_strategy="oversample", feature_selection="variance_threshold",
        )
        config = build_experiment_config(_action("switch_model", {"model_name": "SVC"}, prep=prep))
        assert config.preprocessing.scaling == "minmax"
        assert config.preprocessing.encoding == "onehot"

    def test_experiment_id_preserved(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "SVC"}))
        assert config.experiment_id == "exp_test"


class TestCanonicalActionExecution:
    def test_add_preprocessing_updates_preprocessing_config(self):
        prep = PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard",
            encoding="onehot", imbalance_strategy="none", feature_selection="none",
        )
        # Action with strategy="oversample"
        config = build_experiment_config(
            _action("add_preprocessing", {"model_name": "RandomForestClassifier", "strategy": "oversample"}, prep=prep)
        )
        assert config.preprocessing.imbalance_strategy == "oversample"

    def test_change_optimizer_raises_on_classical_model(self):
        with pytest.raises(ValueError, match="only supported for deep learning"):
            build_experiment_config(
                _action("change_optimizer", {"model_name": "RandomForestClassifier", "learning_rate_scale": 0.1})
            )

    def test_unfreeze_backbone_raises_on_classical_model(self):
        with pytest.raises(ValueError, match="only supported for deep learning"):
            build_experiment_config(
                _action("unfreeze_backbone", {"model_name": "LogisticRegression", "n_layers": 1})
            )

    def test_switch_architecture_raises_on_classical_model(self):
        with pytest.raises(ValueError, match="only supported for deep learning"):
            build_experiment_config(
                _action("switch_architecture", {"model_name": "SVC", "new_arch": "CNN1D"})
            )

    @pytest.mark.parametrize("model_name", [
        "RandomForestClassifier",
        "LogisticRegression",
        "SVC",
        "GradientBoostingRegressor",
        "DecisionTreeClassifier",
    ])
    def test_early_stop_raises_on_classical_models(self, model_name):
        with pytest.raises(ValueError, match="only supported for deep learning"):
            build_experiment_config(
                _action("early_stop", {"model_name": model_name})
            )

    def test_classical_models_have_early_stopping_false(self):
        config = build_experiment_config(
            _action("switch_model", {"model_name": "RandomForestClassifier"})
        )
        assert config.early_stopping is False


class TestHyperparameterMutationDeterminism:
    @pytest.mark.parametrize("model_name,param,higher_is_more_capacity", [
        ("RandomForestClassifier", "n_estimators", True),
        ("GradientBoostingClassifier", "n_estimators", True),
        ("DecisionTreeClassifier", "max_depth", True),
        ("LogisticRegression", "C", True),
        ("SVC", "C", True),
        ("KNeighborsClassifier", "n_neighbors", False),
    ])
    def test_capacity_increase_and_decrease(self, model_name, param, higher_is_more_capacity):
        # 1. Increase capacity
        inc_config = build_experiment_config(
            _action("increase_model_capacity", {"model_name": model_name, "scale": 1.5})
        )
        val_inc = inc_config.hyperparameters.get(param)
        assert val_inc is not None, f"Model {model_name} missing {param} after increase_capacity"

        # 2. Decrease capacity
        dec_config = build_experiment_config(
            _action("decrease_model_capacity", {"model_name": model_name, "scale": 0.75})
        )
        val_dec = dec_config.hyperparameters.get(param)
        assert val_dec is not None, f"Model {model_name} missing {param} after decrease_capacity"

        # 3. Direction and difference check
        assert val_inc != val_dec
        if higher_is_more_capacity:
            assert val_inc > val_dec, f"{model_name}: expected val_inc ({val_inc}) > val_dec ({val_dec}) for {param}"
        else:
            assert val_inc < val_dec, f"{model_name}: expected val_inc ({val_inc}) < val_dec ({val_dec}) for {param}"


class TestUnsupportedModelsRejectCapacity:
    def test_unsupported_models_raise_on_capacity_mutations(self):
        for unsupported in ["LinearRegression", "LinearDiscriminantAnalysis", "UnknownModel"]:
            with pytest.raises(ValueError, match="does not support capacity mutations"):
                build_experiment_config(
                    _action("increase_model_capacity", {"model_name": unsupported})
                )
            with pytest.raises(ValueError, match="does not support capacity mutations"):
                build_experiment_config(
                    _action("decrease_model_capacity", {"model_name": unsupported})
                )


class TestCanonicalModelSpaceConsistency:
    def test_all_default_models_in_registry(self):
        for m in _DEFAULT_MODELS:
            assert m in MODEL_REGISTRY, f"Default classifier '{m}' is not in MODEL_REGISTRY"
        for m in _DEFAULT_REGRESSION_MODELS:
            assert m in MODEL_REGISTRY, f"Default regressor '{m}' is not in MODEL_REGISTRY"

    @pytest.mark.parametrize("model_name", _DEFAULT_MODELS + _DEFAULT_REGRESSION_MODELS)
    def test_default_models_build_valid_ml_config(self, model_name):
        cfg = build_experiment_config(_action("switch_model", {"model_name": model_name}))
        assert cfg.model_name == model_name
        assert cfg.model_type == "ml"
        assert cfg.early_stopping is False

    @pytest.mark.parametrize("model_name", _DEFAULT_MODELS + _DEFAULT_REGRESSION_MODELS)
    def test_classical_model_space_rejects_dl_actions(self, model_name):
        for dl_act in ["change_optimizer", "unfreeze_backbone", "switch_architecture", "early_stop"]:
            with pytest.raises(ValueError, match="only supported for deep learning"):
                build_experiment_config(_action(dl_act, {"model_name": model_name}))


class TestStrengthenedMutationToEstimator:
    @pytest.mark.parametrize("model_name,action_type,param,expected_change_fn", [
        # Capacity mutations (scale=1.5 for increase, scale=0.75 for decrease)
        ("RandomForestClassifier", "increase_model_capacity", "n_estimators", lambda b, a: a > b),
        ("RandomForestClassifier", "decrease_model_capacity", "n_estimators", lambda b, a: a < b),
        ("GradientBoostingClassifier", "increase_model_capacity", "n_estimators", lambda b, a: a > b),
        ("GradientBoostingClassifier", "decrease_model_capacity", "n_estimators", lambda b, a: a < b),
        ("DecisionTreeClassifier", "increase_model_capacity", "max_depth", lambda b, a: a > b),
        ("DecisionTreeClassifier", "decrease_model_capacity", "max_depth", lambda b, a: a < b),
        ("LogisticRegression", "increase_model_capacity", "C", lambda b, a: a > b),
        ("LogisticRegression", "decrease_model_capacity", "C", lambda b, a: a < b),
        ("SVC", "increase_model_capacity", "C", lambda b, a: a > b),
        ("SVC", "decrease_model_capacity", "C", lambda b, a: a < b),
        ("KNeighborsClassifier", "increase_model_capacity", "n_neighbors", lambda b, a: a < b),
        ("KNeighborsClassifier", "decrease_model_capacity", "n_neighbors", lambda b, a: a > b),
        ("Ridge", "increase_model_capacity", "alpha", lambda b, a: a < b),
        ("Ridge", "decrease_model_capacity", "alpha", lambda b, a: a > b),
        ("Lasso", "increase_model_capacity", "alpha", lambda b, a: a < b),
        ("Lasso", "decrease_model_capacity", "alpha", lambda b, a: a > b),
        # Regularization mutations (direction: increase -> stronger regularization)
        ("LogisticRegression", "modify_regularization", "C", lambda b, a: a < b),
        ("SVC", "modify_regularization", "C", lambda b, a: a < b),
        ("Ridge", "modify_regularization", "alpha", lambda b, a: a > b),
        ("Lasso", "modify_regularization", "alpha", lambda b, a: a > b),
        ("RandomForestClassifier", "modify_regularization", "max_depth", lambda b, a: a < b),
        ("GradientBoostingClassifier", "modify_regularization", "max_depth", lambda b, a: a < b),
        ("DecisionTreeClassifier", "modify_regularization", "max_depth", lambda b, a: a < b),
        ("KNeighborsClassifier", "modify_regularization", "n_neighbors", lambda b, a: a > b),
    ])
    def test_before_action_after_estimator_chain(self, model_name, action_type, param, expected_change_fn):
        cls = MODEL_REGISTRY[model_name]
        valid_params = inspect.signature(cls.__init__).parameters
        assert param in valid_params, f"{param} is not in {cls.__name__}.__init__ signature"

        # 1. Before: baseline config and estimator
        base_cfg = build_experiment_config(_action("switch_model", {"model_name": model_name}))
        base_hp = {k: v for k, v in base_cfg.hyperparameters.items() if k in valid_params}
        if param == "max_depth" and "max_depth" not in base_hp:
            # scikit-learn tree/forest models default to max_depth=None.
            # Set explicit baseline depth of 10 so before vs after comparisons are numeric.
            base_hp["max_depth"] = 10
        base_est = cls(**base_hp)
        val_before = getattr(base_est, param)

        # 2. Action: perform hyperparameter mutation
        params = {"model_name": model_name}
        if param == "max_depth":
            params["max_depth"] = val_before
        if action_type == "increase_model_capacity":
            params["scale"] = 1.5
        elif action_type == "decrease_model_capacity":
            params["scale"] = 0.75
        elif action_type == "modify_regularization":
            params["direction"] = "increase"
        mut_action = _action(action_type, params)

        # 3. After: config builder creates updated ExperimentConfig
        after_cfg = build_experiment_config(mut_action)
        val_after_cfg = after_cfg.hyperparameters.get(param)
        assert val_after_cfg is not None, f"Config missing mutated param '{param}' for {model_name}"
        assert expected_change_fn(val_before, val_after_cfg), (
            f"Expected change failed for {model_name}.{param}: before={val_before}, after={val_after_cfg}"
        )

        # 4. Actual estimator config: instantiate scikit-learn model and assert parameter matches exactly
        mut_hp = {k: v for k, v in after_cfg.hyperparameters.items() if k in valid_params}
        estimator = cls(**mut_hp)
        val_estimator = getattr(estimator, param)
        assert val_estimator == val_after_cfg, (
            f"Estimator {cls.__name__}.{param} ({val_estimator}) != config value ({val_after_cfg})"
        )

