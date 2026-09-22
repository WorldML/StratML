"""
ml_pipeline.py
--------------
Phase 5 — Train a scikit-learn model and return raw predictions + timing.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stratml.execution.schemas import ExperimentConfig, DataSplit

_PARAM_GRIDS: dict[str, dict] = {
    "RandomForestClassifier":     {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "max_features": ["sqrt", "log2"]},
    "RandomForestRegressor":      {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "max_features": ["sqrt", "log2"]},
    "ExtraTreesClassifier":       {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "ExtraTreesRegressor":        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "GradientBoostingClassifier": {"learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]},
    "GradientBoostingRegressor":  {"learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]},
    "LogisticRegression":         {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]},
    "SVC":                        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    "SVR":                        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    "Ridge":                      {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "Lasso":                      {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "KNeighborsClassifier":       {"n_neighbors": [3, 5, 7, 11]},
    "KNeighborsRegressor":        {"n_neighbors": [3, 5, 7, 11]},
    "DecisionTreeClassifier":     {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "DecisionTreeRegressor":      {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
}

MODEL_REGISTRY: dict = {
    # ── Linear ────────────────────────────────────────────────────────────────
    "LinearRegression":         LinearRegression,
    "Ridge":                    Ridge,
    "Lasso":                    Lasso,
    "ElasticNet":               ElasticNet,
    "SGDClassifier":            SGDClassifier,
    "SGDRegressor":             SGDRegressor,
    # ── Tree ──────────────────────────────────────────────────────────────────
    "DecisionTreeClassifier":   DecisionTreeClassifier,
    "DecisionTreeRegressor":    DecisionTreeRegressor,
    # ── Ensemble ──────────────────────────────────────────────────────────────
    "LogisticRegression":       LogisticRegression,
    "RandomForestClassifier":   RandomForestClassifier,
    "RandomForestRegressor":    RandomForestRegressor,
    "ExtraTreesClassifier":     ExtraTreesClassifier,
    "ExtraTreesRegressor":      ExtraTreesRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor":  GradientBoostingRegressor,
    "AdaBoostClassifier":       AdaBoostClassifier,
    "AdaBoostRegressor":        AdaBoostRegressor,
    # ── SVM ───────────────────────────────────────────────────────────────────
    "SVC":                      SVC,
    "SVR":                      SVR,
    # ── Neighbors ─────────────────────────────────────────────────────────────
    "KNeighborsClassifier":     KNeighborsClassifier,
    "KNeighborsRegressor":      KNeighborsRegressor,
    # ── Probabilistic ─────────────────────────────────────────────────────────
    "GaussianNB":               GaussianNB,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
}

PAPER_CLASSIFICATION_MODELS: list[str] = [
    "RandomForestClassifier",
    "LogisticRegression",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
    "SVC",
    "KNeighborsClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
]

PAPER_REGRESSION_MODELS: list[str] = [
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "ExtraTreesRegressor",
    "DecisionTreeRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "KNeighborsRegressor",
]

PAPER_CLASSICAL_MODELS: list[str] = PAPER_CLASSIFICATION_MODELS + PAPER_REGRESSION_MODELS


@dataclass
class MLPipelineResult:
    model: object
    y_val_pred: np.ndarray
    train_curve: list[float]
    val_curve: list[float]
    runtime: float


def run_ml_pipeline(config: ExperimentConfig, data_split: DataSplit) -> MLPipelineResult:
    """Instantiate, train (optionally tune), and evaluate a sklearn model."""
    cls = MODEL_REGISTRY.get(config.model_name)
    if cls is None:
        raise ValueError(f"Unknown model '{config.model_name}'. Available: {list(MODEL_REGISTRY)}")

    valid_params = inspect.signature(cls.__init__).parameters
    hp = {k: v for k, v in config.hyperparameters.items() if k in valid_params}
    seed = getattr(config, "seed", 42)
    if "random_state" in valid_params and "random_state" not in hp:
        hp["random_state"] = seed

    t0 = time.perf_counter()

    if config.tune and config.model_name in _PARAM_GRIDS:
        from sklearn.model_selection import RandomizedSearchCV
        base = cls(**hp)
        search = RandomizedSearchCV(
            base,
            _PARAM_GRIDS[config.model_name],
            n_iter=10,
            cv=3,
            random_state=seed,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(data_split.X_train, data_split.y_train)
        model = search.best_estimator_
    else:
        model = cls(**hp)
        model.fit(data_split.X_train, data_split.y_train)

    runtime = round(time.perf_counter() - t0, 4)
    y_val_pred = model.predict(data_split.X_val)

    try:
        from sklearn.metrics import log_loss
        train_loss = round(float(log_loss(data_split.y_train, model.predict_proba(data_split.X_train))), 6)
        val_loss   = round(float(log_loss(data_split.y_val,   model.predict_proba(data_split.X_val))),   6)
    except Exception:
        train_loss = 0.0
        val_loss   = 0.0

    return MLPipelineResult(
        model=model,
        y_val_pred=y_val_pred,
        train_curve=[train_loss],
        val_curve=[val_loss],
        runtime=runtime,
    )
