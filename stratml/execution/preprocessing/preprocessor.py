"""
preprocessor.py
---------------
Phase 4b — Apply preprocessing decisions to DataSplit.

Transforms X_train/X_val according to PreprocessingConfig and DataProfile.
Holdout boundary: X_test/y_test are ONLY transformed when transform_test=True (at final evaluation).
During experimentation loop (transform_test=False), holdout test split is never transformed or accessed.
Returns a new DataSplit with cleaned DataFrames and the config actually applied.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from stratml.execution.schemas import DataProfile, DataSplit, PreprocessingConfig


def apply_preprocessing(
    data_split: DataSplit,
    config: PreprocessingConfig,
    profile: DataProfile,
    transform_test: bool = False,
    seed: int = 42,
) -> tuple[DataSplit, PreprocessingConfig]:
    """
    Apply preprocessing steps to DataSplit.
    When transform_test=False, X_test and y_test are untouched.
    Returns (clean_split, config_applied).
    """
    X_train = data_split.X_train.copy()
    X_val   = data_split.X_val.copy()
    y_train = data_split.y_train.copy()
    y_val   = data_split.y_val.copy()

    if transform_test:
        X_test = data_split.X_test.copy()
        y_test = data_split.y_test.copy()
    else:
        X_test = data_split.X_test
        y_test = data_split.y_test

    num_cols = [c for c in profile.numerical_columns if c in X_train.columns]
    cat_cols = [c for c in profile.categorical_columns if c in X_train.columns]

    # ── 1. Missing value imputation ──────────────────────────────────────────
    if config.missing_value_strategy == "drop":
        mask = X_train.notna().all(axis=1)
        X_train, y_train = X_train[mask].reset_index(drop=True), y_train[mask].reset_index(drop=True)
        mask_val = X_val.notna().all(axis=1)
        X_val, y_val = X_val[mask_val].reset_index(drop=True), y_val[mask_val].reset_index(drop=True)
        if transform_test:
            mask_test = X_test.notna().all(axis=1)
            X_test, y_test = X_test[mask_test].reset_index(drop=True), y_test[mask_test].reset_index(drop=True)
    else:
        strategy = config.missing_value_strategy  # mean | median | mode→most_frequent
        num_strategy = "most_frequent" if strategy == "mode" else strategy
        if num_cols:
            imp = SimpleImputer(strategy=num_strategy)
            X_train[num_cols] = imp.fit_transform(X_train[num_cols])
            X_val[num_cols]   = imp.transform(X_val[num_cols])
            if transform_test:
                X_test[num_cols] = imp.transform(X_test[num_cols])
        if cat_cols:
            imp_cat = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
            X_val[cat_cols]   = imp_cat.transform(X_val[cat_cols])
            if transform_test:
                X_test[cat_cols]  = imp_cat.transform(X_test[cat_cols])

    # ── 2. Categorical encoding ──────────────────────────────────────────────
    if cat_cols and config.encoding != "none":
        if config.encoding == "onehot":
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(X_train[cat_cols])
            new_cols = enc.get_feature_names_out(cat_cols).tolist()

            def _encode_split(split_X):
                encoded = pd.DataFrame(
                    enc.transform(split_X[cat_cols]),
                    columns=new_cols, index=split_X.index,
                )
                return pd.concat(
                    [split_X.drop(columns=cat_cols), encoded], axis=1
                )

            X_train = _encode_split(X_train)
            X_val   = _encode_split(X_val)
            if transform_test:
                X_test  = _encode_split(X_test)
        elif config.encoding == "label":
            for col in cat_cols:
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))
                known = set(le.classes_)
                X_train[col] = le.transform(X_train[col].astype(str))
                X_val[col]   = X_val[col].astype(str).map(lambda v: le.transform([v])[0] if v in known else -1)
                if transform_test:
                    X_test[col]  = X_test[col].astype(str).map(lambda v: le.transform([v])[0] if v in known else -1)

    # ── 3. Numerical scaling ─────────────────────────────────────────────────
    # Recompute after encoding — onehot may have dropped original cat cols
    active_num_cols = [c for c in num_cols if c in X_train.columns]
    if active_num_cols and config.scaling != "none":
        if config.scaling == "standard":
            scaler = StandardScaler()
        elif config.scaling == "robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
        X_train[active_num_cols] = scaler.fit_transform(X_train[active_num_cols])
        X_val[active_num_cols]   = scaler.transform(X_val[active_num_cols])
        if transform_test:
            X_test[active_num_cols]  = scaler.transform(X_test[active_num_cols])

    # ── 4. Imbalance correction (train only) ─────────────────────────────────
    if config.imbalance_strategy != "none":
        try:
            if config.imbalance_strategy == "oversample":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=seed)
            else:
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=seed)
            X_train_arr, y_train_arr = sampler.fit_resample(X_train, y_train)
            X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
            y_train = pd.Series(y_train_arr, name=y_train.name)
        except ImportError:
            import warnings
            warnings.warn(
                "imbalanced-learn is not installed — imbalance correction skipped. "
                "Install with: pip install imbalanced-learn",
                UserWarning,
                stacklevel=2,
            )

    # ── 5. Feature selection ─────────────────────────────────────────────────
    if config.feature_selection == "variance_threshold":
        sel = VarianceThreshold()
        sel.fit(X_train)
        mask = sel.get_support()
        kept = X_train.columns[mask].tolist()
        X_train = X_train[kept]
        X_val   = X_val[kept]
        if transform_test:
            X_test  = X_test[kept]

    return DataSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    ), config
