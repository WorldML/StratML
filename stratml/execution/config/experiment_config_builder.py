"""
experiment_config_builder.py
-----------------------------
Phase 4 — Translate ActionDecision → ExperimentConfig.

Mutation logic lives in ml_mutations.py and dl_mutations.py.
This file is a pure dispatcher — one function, no mutation logic.
"""

from __future__ import annotations

import logging
from stratml.execution.schemas import ActionDecision, ExperimentConfig, PreprocessingConfig
from stratml.execution.config import ml_mutations, dl_mutations

log = logging.getLogger(__name__)

_DL_MODELS = {
    # Tabular
    "MLP", "CNN1D", "RNN", "ResidualMLP", "TabTransformer", "PyTorchMLP",
    # Vision
    "CNN2D", "ResNet18", "EfficientNetB0", "MobileNetV3",
    # Text
    "TextCNN", "BiLSTM", "DistilBERT", "TinyBERT",
}

_DL_VISION_MODELS   = ["CNN2D", "ResNet18", "EfficientNetB0", "MobileNetV3"]
_DL_TEXT_MODELS     = ["TextCNN", "BiLSTM", "DistilBERT", "TinyBERT"]
_DL_TABULAR_MODELS  = ["MLP", "CNN1D", "RNN", "ResidualMLP", "TabTransformer"]


def build_experiment_config(action: ActionDecision, tune: bool = False, seed: int = 42) -> ExperimentConfig:
    """Build an executable ExperimentConfig from an ActionDecision."""
    params      = dict(action.parameters)
    action_type = action.action_type
    model_name  = params.pop("model_name", "LogisticRegression")
    hp          = dict(params)
    is_dl       = model_name in _DL_MODELS
    preprocessing = action.preprocessing

    if action_type == "switch_model":
        hp = {}

    elif action_type == "modify_regularization":
        direction = hp.pop("direction", "increase")
        hp = dl_mutations.mutate_regularization(hp, direction) if is_dl \
            else ml_mutations.mutate_regularization(model_name, hp, direction)

    elif action_type == "increase_model_capacity":
        scale = float(hp.pop("scale", 1.5))
        hp = dl_mutations.increase_capacity(hp, scale) if is_dl \
            else ml_mutations.increase_capacity(model_name, hp, scale)

    elif action_type == "decrease_model_capacity":
        scale = float(hp.pop("scale", 0.75))
        hp = dl_mutations.decrease_capacity(hp, scale) if is_dl \
            else ml_mutations.decrease_capacity(model_name, hp, scale)

    elif action_type == "change_optimizer":
        if not is_dl:
            raise ValueError(
                f"Action 'change_optimizer' is only supported for deep learning models, but model is classical: '{model_name}'"
            )
        lr_scale = float(hp.pop("learning_rate_scale", 0.1))
        hp = dl_mutations.mutate_optimizer(hp, lr_scale)

    elif action_type == "unfreeze_backbone":
        if not is_dl:
            raise ValueError(
                f"Action 'unfreeze_backbone' is only supported for deep learning models, but model is classical: '{model_name}'"
            )
        n_layers = int(hp.pop("n_layers", 1))
        hp = dl_mutations.unfreeze_backbone(hp, n_layers)

    elif action_type == "switch_architecture":
        if not is_dl:
            raise ValueError(
                f"Action 'switch_architecture' is only supported for deep learning models, but model is classical: '{model_name}'"
            )
        new_arch = hp.pop("new_arch", hp.get("architecture", "MLP"))
        hp = dl_mutations.switch_architecture(hp, new_arch)

    elif action_type in ("add_preprocessing", "apply_preprocessing"):
        prep_dict = preprocessing.model_dump()
        strat = params.get("strategy")
        if strat in ("oversample", "undersample", "none"):
            prep_dict["imbalance_strategy"] = strat
        elif strat in ("standard", "minmax", "robust"):
            prep_dict["scaling"] = strat
        for k in ("missing_value_strategy", "scaling", "encoding", "imbalance_strategy", "feature_selection"):
            if k in params:
                prep_dict[k] = params[k]
        preprocessing = PreprocessingConfig(**prep_dict)

    elif action_type == "early_stop":
        if not is_dl:
            raise ValueError(
                f"Action 'early_stop' is only supported for deep learning models, but model is classical: '{model_name}'"
            )

    elif action_type == "terminate":
        pass

    else:
        raise ValueError(f"Unknown action_type: '{action_type}'")

    model_type = "dl" if is_dl else "ml"
    early_stopping = True if is_dl else False
    patience = int(params.get("early_stopping_patience", 5))

    return ExperimentConfig(
        experiment_id=action.experiment_id,
        model_name=model_name,
        model_type=model_type,
        hyperparameters=hp,
        preprocessing=preprocessing,
        early_stopping=early_stopping,
        early_stopping_patience=patience,
        tune=tune and not is_dl,
        seed=seed,
    )
