"""
conftest.py
-----------
Shared pytest fixtures for all execution team tests.
"""

import numpy as np
import pandas as pd
import pytest

from stratml.execution.schemas import (
    ActionDecision, DataProfile, DataSplit, Dataset,
    ExperimentConfig, FeatureInfo, PreprocessingConfig, SplitConfig,
)


# ── Global offline fixture ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def disable_external_apis(monkeypatch):
    """Ensure test suite runs deterministically and offline without hitting external APIs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


# ── Reusable PreprocessingConfig ─────────────────────────────────────────────

@pytest.fixture
def default_prep():
    return PreprocessingConfig(
        missing_value_strategy="mean",
        scaling="standard",
        encoding="none",
        imbalance_strategy="none",
        feature_selection="none",
    )


# ── Synthetic classification dataset (100 rows, 4 features, 3 classes) ───────

@pytest.fixture
def clf_df():
    np.random.seed(0)
    n = 120
    df = pd.DataFrame({
        "f0": np.random.randn(n),
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "target": np.tile(["a", "b", "c"], n // 3),
    })
    return df


@pytest.fixture
def clf_dataset(clf_df):
    return Dataset(
        dataset_name="test_clf",
        rows=len(clf_df),
        columns=len(clf_df.columns),
        target_column="target",
        dataset_type="tabular",
        raw_dataframe=clf_df,
    )


@pytest.fixture
def clf_profile(clf_df):
    return DataProfile(
        dataset_name="test_clf",
        dataset_type="tabular",
        rows=len(clf_df),
        columns=len(clf_df.columns),
        target_column="target",
        problem_type="classification",
        numerical_columns=["f0", "f1", "f2", "f3"],
        categorical_columns=[],
        missing_value_ratio=0.0,
        class_distribution={"a": 40, "b": 40, "c": 40},
        feature_summary=[
            FeatureInfo(name=c, dtype="float64", unique_values=120, missing_percentage=0.0, distribution="normal")
            for c in ["f0", "f1", "f2", "f3"]
        ],
        recommended_metrics=["accuracy", "f1_score"],
    )


# ── Synthetic regression dataset (100 rows, 4 features) ──────────────────────

@pytest.fixture
def reg_df():
    np.random.seed(1)
    n = 100
    df = pd.DataFrame({
        "f0": np.random.randn(n),
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "target": np.random.randn(n).astype(np.float32),
    })
    return df


@pytest.fixture
def reg_dataset(reg_df):
    return Dataset(
        dataset_name="test_reg",
        rows=len(reg_df),
        columns=len(reg_df.columns),
        target_column="target",
        dataset_type="tabular",
        raw_dataframe=reg_df,
    )


@pytest.fixture
def reg_profile(reg_df):
    return DataProfile(
        dataset_name="test_reg",
        dataset_type="tabular",
        rows=len(reg_df),
        columns=len(reg_df.columns),
        target_column="target",
        problem_type="regression",
        numerical_columns=["f0", "f1", "f2", "f3"],
        categorical_columns=[],
        missing_value_ratio=0.0,
        class_distribution={},
        feature_summary=[
            FeatureInfo(name=c, dtype="float64", unique_values=100, missing_percentage=0.0, distribution="normal")
            for c in ["f0", "f1", "f2", "f3"]
        ],
        recommended_metrics=["mse", "rmse", "r2"],
    )


# ── Reusable DataSplit (classification) ───────────────────────────────────────

@pytest.fixture
def clf_split(clf_df):
    X = clf_df.drop(columns=["target"])
    y = clf_df["target"]
    return DataSplit(
        X_train=X.iloc[:84].reset_index(drop=True),
        X_val=X.iloc[84:96].reset_index(drop=True),
        X_test=X.iloc[96:].reset_index(drop=True),
        y_train=y.iloc[:84].reset_index(drop=True),
        y_val=y.iloc[84:96].reset_index(drop=True),
        y_test=y.iloc[96:].reset_index(drop=True),
    )


@pytest.fixture
def reg_split(reg_df):
    X = reg_df.drop(columns=["target"])
    y = reg_df["target"]
    return DataSplit(
        X_train=X.iloc[:70].reset_index(drop=True),
        X_val=X.iloc[70:85].reset_index(drop=True),
        X_test=X.iloc[85:].reset_index(drop=True),
        y_train=y.iloc[:70].reset_index(drop=True),
        y_val=y.iloc[70:85].reset_index(drop=True),
        y_test=y.iloc[85:].reset_index(drop=True),
    )


# ── Reusable ActionDecision ───────────────────────────────────────────────────

@pytest.fixture
def base_action(default_prep):
    return ActionDecision(
        experiment_id="test_exp",
        action_type="switch_model",
        parameters={"model_name": "RandomForestClassifier", "n_estimators": 10},
        preprocessing=default_prep,
        reason="test",
        expected_gain=0.0,
        expected_cost=1.0,
        confidence=1.0,
    )


@pytest.fixture
def base_config(default_prep):
    return ExperimentConfig(
        experiment_id="test_exp",
        model_name="RandomForestClassifier",
        model_type="ml",
        hyperparameters={"n_estimators": 10},
        preprocessing=default_prep,
    )
