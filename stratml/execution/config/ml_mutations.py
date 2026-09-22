"""
ml_mutations.py
---------------
Authoritative hyperparameter mutation logic for ML (sklearn) models.
Called by experiment_config_builder.py for ML action types.
"""

from __future__ import annotations

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


def mutate_regularization(model_name: str, hp: dict, direction: str) -> dict:
    """Return updated hyperparams with regularization adjusted for the given model."""
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
    return hp_out


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
