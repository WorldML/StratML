"""
ml_mutations.py
---------------
Authoritative hyperparameter mutation logic for ML (sklearn) models.
Called by experiment_config_builder.py for ML action types.
"""

from __future__ import annotations

import copy
from typing import Any

# (param_name, default_value, delta_when_increasing_regularization)
_REG_PARAM: dict[str, tuple[str, float, float]] = {
    "LogisticRegression": ("C",             1.0,   0.1),
    "SVC":                ("C",             1.0,   0.1),
    "SVR":                ("C",             1.0,   0.1),
    "Ridge":              ("alpha",         1.0,   10.0),
    "Lasso":              ("alpha",         1.0,   10.0),
    "ElasticNet":         ("alpha",         1.0,   10.0),
    "RandomForest":       ("max_depth",     10,    -2),
    "GradientBoosting":   ("max_depth",      3,    -1),
    "ExtraTrees":         ("max_depth",     10,    -2),
    "DecisionTree":       ("max_depth",     10,    -2),
    "KNeighbors":         ("n_neighbors",    5,     2),
    "GaussianNB":         ("var_smoothing", 1e-9,  10.0),
}


# ===========================================================================
# Frozen Machine-Readable Specification of StratML Hyperparameter Mutation Space
# (Distinct from RandomizedSearchCV tuning space grids)
# paper model -> mutable parameter -> mutation/action -> legal domain/range
# ===========================================================================

_TREE_ENSEMBLE_SPEC: dict[str, Any] = {
    "max_depth": {
        "param_type": "int",
        "domain": {"min": 1, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 10,
                "delta_on_increase": -2,
                "delta_on_decrease": 2,
            },
            "increase_model_capacity": {
                "scale": 1.5,
                "default": 10,
            },
            "decrease_model_capacity": {
                "scale": 0.75,
                "min_val": 2,
                "default": 10,
            },
        },
    },
    "n_estimators": {
        "param_type": "int",
        "domain": {"min": 10, "max": None},
        "supported_actions": {
            "increase_model_capacity": {
                "scale": 1.5,
                "default": 100,
            },
            "decrease_model_capacity": {
                "scale": 0.75,
                "min_val": 10,
                "default": 100,
            },
        },
    },
}

_GB_SPEC: dict[str, Any] = {
    "max_depth": {
        "param_type": "int",
        "domain": {"min": 1, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 3,
                "delta_on_increase": -1,
                "delta_on_decrease": 2,
            },
            "increase_model_capacity": {
                "step": 1,
                "default": 3,
            },
            "decrease_model_capacity": {
                "step": -1,
                "min_val": 1,
                "default": 3,
            },
        },
    },
    "n_estimators": {
        "param_type": "int",
        "domain": {"min": 10, "max": None},
        "supported_actions": {
            "increase_model_capacity": {
                "scale": 1.5,
                "default": 100,
            },
            "decrease_model_capacity": {
                "scale": 0.75,
                "min_val": 10,
                "default": 100,
            },
        },
    },
}

_DT_SPEC: dict[str, Any] = {
    "max_depth": {
        "param_type": "int",
        "domain": {"min": 1, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 10,
                "delta_on_increase": -2,
                "delta_on_decrease": 2,
            },
            "increase_model_capacity": {
                "scale": 1.5,
                "default": 5,
            },
            "decrease_model_capacity": {
                "scale": 0.75,
                "min_val": 1,
                "default": 5,
            },
        },
    },
}

_C_SPEC: dict[str, Any] = {
    "C": {
        "param_type": "float",
        "domain": {"min": 0.001, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 1.0,
                "factor_on_increase": 0.1,
                "factor_on_decrease": 10.0,
            },
            "increase_model_capacity": {
                "scale": 1.5,
                "default": 1.0,
            },
            "decrease_model_capacity": {
                "scale": 0.75,
                "min_val": 0.001,
                "default": 1.0,
            },
        },
    },
}

_ALPHA_SPEC: dict[str, Any] = {
    "alpha": {
        "param_type": "float",
        "domain": {"min": 0.0001, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 1.0,
                "factor_on_increase": 10.0,
                "factor_on_decrease": 0.1,
            },
            "increase_model_capacity": {
                "scale_divisor": 1.5,
                "min_val": 0.0001,
                "default": 1.0,
            },
            "decrease_model_capacity": {
                "scale_divisor": 0.75,
                "default": 1.0,
            },
        },
    },
}

_KNN_SPEC: dict[str, Any] = {
    "n_neighbors": {
        "param_type": "int",
        "domain": {"min": 1, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 5,
                "delta_on_increase": 2,
                "delta_on_decrease": -2,
            },
            "increase_model_capacity": {
                "scale_divisor": 1.5,
                "min_val": 1,
                "default": 5,
            },
            "decrease_model_capacity": {
                "scale_divisor": 0.75,
                "default": 5,
            },
        },
    },
}

_GNB_SPEC: dict[str, Any] = {
    "var_smoothing": {
        "param_type": "float",
        "domain": {"min": 1e-12, "max": None},
        "supported_actions": {
            "modify_regularization": {
                "directions": ["increase", "decrease"],
                "default": 1e-9,
                "factor_on_increase": 10.0,
                "factor_on_decrease": 0.1,
            },
            "increase_model_capacity": {
                "scale_divisor": 1.5,
                "min_val": 1e-12,
                "default": 1e-9,
            },
            "decrease_model_capacity": {
                "scale_divisor": 0.75,
                "default": 1e-9,
            },
        },
    },
}

PAPER_MUTATION_SPACE: dict[str, dict[str, Any]] = {
    # ── Paper Classification Models (8) ───────────────────────────────────────
    "RandomForestClassifier":     copy.deepcopy(_TREE_ENSEMBLE_SPEC),
    "LogisticRegression":         copy.deepcopy(_C_SPEC),
    "GradientBoostingClassifier": copy.deepcopy(_GB_SPEC),
    "ExtraTreesClassifier":       copy.deepcopy(_TREE_ENSEMBLE_SPEC),
    "SVC":                        copy.deepcopy(_C_SPEC),
    "KNeighborsClassifier":       copy.deepcopy(_KNN_SPEC),
    "GaussianNB":                 copy.deepcopy(_GNB_SPEC),
    "DecisionTreeClassifier":     copy.deepcopy(_DT_SPEC),

    # ── Paper Regression Models (8) ───────────────────────────────────────────
    "RandomForestRegressor":      copy.deepcopy(_TREE_ENSEMBLE_SPEC),
    "GradientBoostingRegressor":  copy.deepcopy(_GB_SPEC),
    "ExtraTreesRegressor":        copy.deepcopy(_TREE_ENSEMBLE_SPEC),
    "DecisionTreeRegressor":      copy.deepcopy(_DT_SPEC),
    "Ridge":                      copy.deepcopy(_ALPHA_SPEC),
    "Lasso":                      copy.deepcopy(_ALPHA_SPEC),
    "ElasticNet":                 copy.deepcopy(_ALPHA_SPEC),
    "KNeighborsRegressor":        copy.deepcopy(_KNN_SPEC),
}


def get_paper_mutation_space() -> dict[str, dict[str, Any]]:
    """Return an isolated copy of the frozen paper hyperparameter mutation space."""
    return copy.deepcopy(PAPER_MUTATION_SPACE)


def get_model_mutation_spec(model_name: str) -> dict[str, Any]:
    """Inspect the frozen mutation specification for a specific paper model."""
    if model_name not in PAPER_MUTATION_SPACE:
        raise ValueError(
            f"Model '{model_name}' is not in the frozen paper classical model mutation space. "
            f"Available paper models: {sorted(PAPER_MUTATION_SPACE.keys())}"
        )
    return copy.deepcopy(PAPER_MUTATION_SPACE[model_name])


def is_mutation_supported(model_name: str, action_type: str, param: str | None = None) -> bool:
    """Return True if model, action, and optional parameter are in the supported mutation space."""
    if model_name not in PAPER_MUTATION_SPACE:
        return False
    model_spec = PAPER_MUTATION_SPACE[model_name]
    if param is not None:
        if param not in model_spec:
            return False
        return action_type in model_spec[param]["supported_actions"]
    return any(action_type in param_info["supported_actions"] for param_info in model_spec.values())


def mutate_regularization(model_name: str, hp: dict, direction: str) -> dict:
    """Return updated hyperparams with regularization adjusted for the given model."""
    if direction not in ("increase", "decrease"):
        raise ValueError(f"Invalid direction '{direction}'. Must be 'increase' or 'decrease'.")

    hp_out = dict(hp)
    for prefix, (param, default, delta) in _REG_PARAM.items():
        if model_name.startswith(prefix):
            current = hp_out.get(param, default)
            if direction == "increase":
                if param == "C":
                    new_val = round(float(current) * 0.1, 6)
                elif param == "var_smoothing":
                    new_val = round(float(current) * 10.0, 12)
                elif delta < 0:
                    new_val = max(1, int(current) + int(delta))
                elif param == "n_neighbors":
                    new_val = int(current) + int(delta)
                else:
                    new_val = round(float(current) * 10.0, 6)
            else:
                if param == "C":
                    new_val = round(float(current) * 10.0, 6)
                elif param == "var_smoothing":
                    new_val = round(float(current) * 0.1, 12)
                elif delta < 0:
                    new_val = int(current) + 2
                elif param == "n_neighbors":
                    new_val = max(1, int(current) - int(delta))
                else:
                    new_val = round(float(current) * 0.1, 6)
            hp_out[param] = new_val
            return hp_out

    raise ValueError(f"Model '{model_name}' does not support regularization mutations")


def increase_capacity(model_name_or_hp: str | dict, hp: dict | None = None, scale: float = 1.5) -> dict:
    """Increase model capacity (higher model complexity)."""
    if isinstance(model_name_or_hp, dict):
        model_name = ""
        hp_out = dict(model_name_or_hp)
    else:
        model_name = model_name_or_hp
        hp_out = dict(hp or {})

    if model_name.startswith(("RandomForest", "ExtraTrees")):
        n = hp_out.get("n_estimators", 100)
        d = hp_out.get("max_depth", 10)
        hp_out["n_estimators"] = int(n * scale)
        hp_out["max_depth"] = int(d * scale)
    elif model_name.startswith("GradientBoosting"):
        n = hp_out.get("n_estimators", 100)
        d = hp_out.get("max_depth", 3)
        hp_out["n_estimators"] = int(n * scale)
        hp_out["max_depth"] = d + 1
    elif model_name.startswith("DecisionTree"):
        d = hp_out.get("max_depth", 5)
        hp_out["max_depth"] = max(d + 1, int(d * scale))
    elif model_name.startswith(("LogisticRegression", "SVC", "SVR")):
        c = hp_out.get("C", 1.0)
        hp_out["C"] = round(float(c) * scale, 6)
    elif model_name.startswith("KNeighbors"):
        # In k-NN, fewer neighbors = higher capacity (tighter fit / more complex decision boundary)
        n = hp_out.get("n_neighbors", 5)
        hp_out["n_neighbors"] = max(1, int(round(n / scale)))
    elif model_name.startswith("AdaBoost"):
        n = hp_out.get("n_estimators", 50)
        hp_out["n_estimators"] = int(n * scale)
    elif model_name.startswith(("Ridge", "Lasso", "ElasticNet")):
        alpha = hp_out.get("alpha", 1.0)
        hp_out["alpha"] = round(max(0.0001, float(alpha) / scale), 6)
    elif model_name.startswith("GaussianNB"):
        v = hp_out.get("var_smoothing", 1e-9)
        hp_out["var_smoothing"] = round(max(1e-12, float(v) / scale), 12)
    elif model_name.startswith("SGD"):
        alpha = hp_out.get("alpha", 0.0001)
        hp_out["alpha"] = round(max(1e-7, float(alpha) / scale), 7)
    else:
        raise ValueError(f"Model '{model_name}' does not support capacity mutations")

    return hp_out


def decrease_capacity(model_name_or_hp: str | dict, hp: dict | None = None, scale: float = 0.75) -> dict:
    """Decrease model capacity (lower model complexity)."""
    if isinstance(model_name_or_hp, dict):
        model_name = ""
        hp_out = dict(model_name_or_hp)
    else:
        model_name = model_name_or_hp
        hp_out = dict(hp or {})

    if model_name.startswith(("RandomForest", "ExtraTrees")):
        n = hp_out.get("n_estimators", 100)
        d = hp_out.get("max_depth", 10)
        hp_out["n_estimators"] = max(10, int(n * scale))
        hp_out["max_depth"] = max(2, int(d * scale))
    elif model_name.startswith("GradientBoosting"):
        n = hp_out.get("n_estimators", 100)
        d = hp_out.get("max_depth", 3)
        hp_out["n_estimators"] = max(10, int(n * scale))
        hp_out["max_depth"] = max(1, d - 1)
    elif model_name.startswith("DecisionTree"):
        d = hp_out.get("max_depth", 5)
        hp_out["max_depth"] = max(1, int(d * scale))
    elif model_name.startswith(("LogisticRegression", "SVC", "SVR")):
        c = hp_out.get("C", 1.0)
        hp_out["C"] = round(max(0.001, float(c) * scale), 6)
    elif model_name.startswith("KNeighbors"):
        # In k-NN, more neighbors = lower capacity (smoother decision boundary)
        n = hp_out.get("n_neighbors", 5)
        hp_out["n_neighbors"] = int(round(n / scale))
    elif model_name.startswith("AdaBoost"):
        n = hp_out.get("n_estimators", 50)
        hp_out["n_estimators"] = max(10, int(n * scale))
    elif model_name.startswith(("Ridge", "Lasso", "ElasticNet")):
        alpha = hp_out.get("alpha", 1.0)
        hp_out["alpha"] = round(float(alpha) / scale, 6)
    elif model_name.startswith("GaussianNB"):
        v = hp_out.get("var_smoothing", 1e-9)
        hp_out["var_smoothing"] = round(float(v) / scale, 12)
    elif model_name.startswith("SGD"):
        alpha = hp_out.get("alpha", 0.0001)
        hp_out["alpha"] = round(float(alpha) / scale, 7)
    else:
        raise ValueError(f"Model '{model_name}' does not support capacity mutations")

    return hp_out
