"""
test_config_builder.py
----------------------
Unit tests for execution/config/experiment_config_builder.py
"""

import pytest

from stratml.execution.config.experiment_config_builder import build_experiment_config
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

