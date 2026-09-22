"""
test_preprocessor.py
--------------------
Unit tests for execution/preprocessing/preprocessor.py

Key invariant tested throughout: all transformers fit on X_train only.
"""

import numpy as np
import pandas as pd
import pytest

from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.schemas import DataProfile, DataSplit, FeatureInfo, PreprocessingConfig


def _make_split(X, y):
    n = len(X)
    t, v = int(n * 0.7), int(n * 0.85)
    return DataSplit(
        X_train=X.iloc[:t].reset_index(drop=True),
        X_val=X.iloc[t:v].reset_index(drop=True),
        X_test=X.iloc[v:].reset_index(drop=True),
        y_train=y.iloc[:t].reset_index(drop=True),
        y_val=y.iloc[t:v].reset_index(drop=True),
        y_test=y.iloc[v:].reset_index(drop=True),
    )


def _profile(num_cols, cat_cols, n=100):
    all_cols = num_cols + cat_cols
    return DataProfile(
        dataset_name="test", dataset_type="tabular",
        rows=n, columns=len(all_cols) + 1,
        target_column="target", problem_type="classification",
        numerical_columns=num_cols, categorical_columns=cat_cols,
        missing_value_ratio=0.0, class_distribution={"a": 50, "b": 50},
        feature_summary=[
            FeatureInfo(name=c, dtype="float64", unique_values=50, missing_percentage=0.0, distribution="normal")
            for c in all_cols
        ],
        recommended_metrics=["accuracy"],
    )


def _prep(**kwargs):
    defaults = dict(missing_value_strategy="mean", scaling="none",
                    encoding="none", imbalance_strategy="none", feature_selection="none")
    defaults.update(kwargs)
    return PreprocessingConfig(**defaults)


# ── Missing value imputation ──────────────────────────────────────────────────

class TestImputation:
    def test_mean_fills_nan(self):
        np.random.seed(0)
        n = 100
        X = pd.DataFrame({"f0": np.random.randn(n), "f1": np.random.randn(n)})
        X.loc[5, "f0"] = np.nan
        X.loc[10, "f1"] = np.nan
        y = pd.Series(np.tile(["a", "b"], n // 2))
        split = _make_split(X, y)
        profile = _profile(["f0", "f1"], [])

        clean, _ = apply_preprocessing(split, _prep(missing_value_strategy="mean"), profile)
        assert clean.X_train.isnull().sum().sum() == 0
        assert clean.X_val.isnull().sum().sum() == 0

    def test_drop_removes_nan_rows(self):
        n = 100
        X = pd.DataFrame({"f0": np.random.randn(n)})
        X.loc[0, "f0"] = np.nan
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0"], [])

        clean, _ = apply_preprocessing(split, _prep(missing_value_strategy="drop"), profile)
        assert clean.X_train.isnull().sum().sum() == 0

    def test_imputer_not_fit_on_val(self):
        """Imputer fit on train (values≈1) should fill val NaN with ≈1, not val mean (≈100)."""
        np.random.seed(42)
        n = 100
        # train rows 0-69: value=1, val rows 70-84: value=100
        X = pd.DataFrame({"f0": np.concatenate([np.ones(70), np.ones(30) * 100])})
        X.loc[0, "f0"] = np.nan    # NaN in train — will be filled with train mean ≈ 1
        X.loc[70, "f0"] = np.nan   # NaN in val (first val row after reset_index) — should be filled with train mean ≈ 1
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0"], [])

        clean, _ = apply_preprocessing(split, _prep(missing_value_strategy="mean"), profile)
        # Val NaN at index 0 (after reset) should be filled with train mean ≈ 1, not val mean ≈ 100
        assert clean.X_val.loc[0, "f0"] < 10


# ── Scaling ───────────────────────────────────────────────────────────────────

class TestScaling:
    def test_standard_scaler_train_mean_near_zero(self):
        np.random.seed(0)
        n = 100
        X = pd.DataFrame({"f0": np.random.randn(n) * 10 + 50})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0"], [])

        clean, _ = apply_preprocessing(split, _prep(scaling="standard"), profile)
        assert abs(clean.X_train["f0"].mean()) < 0.1

    def test_standard_scaler_train_std_near_one(self):
        np.random.seed(0)
        n = 100
        X = pd.DataFrame({"f0": np.random.randn(n) * 10 + 50})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0"], [])

        clean, _ = apply_preprocessing(split, _prep(scaling="standard"), profile)
        assert abs(clean.X_train["f0"].std() - 1.0) < 0.1

    def test_minmax_scaler_train_range_zero_to_one(self):
        np.random.seed(0)
        n = 100
        X = pd.DataFrame({"f0": np.random.randn(n) * 10 + 50})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0"], [])

        clean, _ = apply_preprocessing(split, _prep(scaling="minmax"), profile)
        assert clean.X_train["f0"].min() >= 0.0
        assert clean.X_train["f0"].max() <= 1.0

    def test_scaler_fit_on_train_only(self):
        """Val values can exceed [0,1] with minmax if val range > train range."""
        np.random.seed(0)
        n = 100
        # Train: values 0-1, Val: values 0-100 (wider range)
        X = pd.DataFrame({"f0": list(np.linspace(0, 1, 70)) + list(np.linspace(0, 100, 30))})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0"], [])

        clean, _ = apply_preprocessing(split, _prep(scaling="minmax"), profile)
        # Val max should be >> 1 because scaler was fit on train (0-1 range)
        assert clean.X_val["f0"].max() > 1.0


# ── Encoding ──────────────────────────────────────────────────────────────────

class TestEncoding:
    def test_onehot_expands_columns(self):
        n = 100
        X = pd.DataFrame({"num": np.random.randn(n), "cat": np.tile(["x", "y", "z"], n // 3 + 1)[:n]})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["num"], ["cat"])

        clean, _ = apply_preprocessing(split, _prep(encoding="onehot"), profile)
        # "cat" column gone, replaced by cat_x, cat_y, cat_z
        assert "cat" not in clean.X_train.columns
        assert clean.X_train.shape[1] > 1

    def test_label_encoding_produces_integers(self):
        n = 90
        X = pd.DataFrame({"num": np.random.randn(n), "cat": np.tile(["x", "y", "z"], n // 3)})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["num"], ["cat"])

        clean, _ = apply_preprocessing(split, _prep(encoding="label"), profile)
        assert clean.X_train["cat"].dtype in [np.int32, np.int64, int]

    def test_none_encoding_leaves_columns_unchanged(self):
        n = 90
        X = pd.DataFrame({"num": np.random.randn(n), "cat": np.tile(["x", "y", "z"], n // 3)})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["num"], ["cat"])

        clean, _ = apply_preprocessing(split, _prep(encoding="none"), profile)
        assert "cat" in clean.X_train.columns


# ── Feature selection ─────────────────────────────────────────────────────────

class TestFeatureSelection:
    def test_variance_threshold_removes_zero_variance_column(self):
        n = 100
        X = pd.DataFrame({
            "f0": np.random.randn(n),
            "const": np.ones(n),   # zero variance — should be removed
        })
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0", "const"], [])

        clean, _ = apply_preprocessing(split, _prep(feature_selection="variance_threshold"), profile)
        assert "const" not in clean.X_train.columns
        assert "f0" in clean.X_train.columns

    def test_feature_selection_applied_to_val_and_test(self):
        n = 100
        X = pd.DataFrame({"f0": np.random.randn(n), "const": np.ones(n)})
        y = pd.Series(["a"] * n)
        split = _make_split(X, y)
        profile = _profile(["f0", "const"], [])

        # Holdout contract: experimentation loop (transform_test=False) does not touch X_test
        clean_loop, _ = apply_preprocessing(
            split, _prep(feature_selection="variance_threshold"), profile, transform_test=False
        )
        assert "const" not in clean_loop.X_val.columns
        assert "const" in clean_loop.X_test.columns

        # Final evaluation call (transform_test=True) transforms X_test
        clean_final, _ = apply_preprocessing(
            split, _prep(feature_selection="variance_threshold"), profile, transform_test=True
        )
        assert "const" not in clean_final.X_val.columns
        assert "const" not in clean_final.X_test.columns


# ── Return value ──────────────────────────────────────────────────────────────

class TestReturnValue:
    def test_returns_preprocessing_config(self, clf_split, clf_profile, default_prep):
        _, applied = apply_preprocessing(clf_split, default_prep, clf_profile)
        assert applied == default_prep

    def test_original_split_not_mutated(self, clf_split, clf_profile, default_prep):
        original_shape = clf_split.X_train.shape
        apply_preprocessing(clf_split, default_prep, clf_profile)
        assert clf_split.X_train.shape == original_shape
