"""
profiler.py
-----------
Phase 2 — Data Profiling: compute a DataProfile from a Dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from stratml.execution.schemas import Dataset, DataProfile, FeatureInfo

_CLASSIFICATION_UNIQUE_THRESHOLD = 20


def build_profile(dataset: Dataset, random_seed: int = 42) -> DataProfile:
    df: pd.DataFrame = dataset.raw_dataframe
    target = dataset.target_column

    feature_cols = [c for c in df.columns if c != target]
    feature_df = df[feature_cols]

    numerical_cols, categorical_cols = _split_column_types(feature_df)
    global_missing = df.isnull().values.mean()
    problem_type = _infer_problem_type(df[target])
    class_distribution = _class_distribution(df[target], problem_type)
    feature_summary = [_describe_feature(df[col], random_seed=random_seed) for col in feature_cols]

    imbalance_ratio = _imbalance_ratio(class_distribution) if problem_type == "classification" else None
    feature_variance_mean = _feature_variance_mean(df[numerical_cols]) if numerical_cols else None
    class_entropy = _class_entropy(class_distribution) if problem_type == "classification" else None

    return DataProfile(
        dataset_name=dataset.dataset_name,
        dataset_type=dataset.dataset_type,
        rows=dataset.rows,
        columns=dataset.columns,
        target_column=target,
        problem_type=problem_type,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        missing_value_ratio=round(float(global_missing), 4),
        class_distribution=class_distribution,
        feature_summary=feature_summary,
        recommended_metrics=_recommend_metrics(problem_type),
        imbalance_ratio=imbalance_ratio,
        feature_variance_mean=feature_variance_mean,
        class_entropy=class_entropy,
        dataset_fingerprint=getattr(dataset, "dataset_fingerprint", None),
    )


def _split_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical, categorical


def _infer_problem_type(target_series: pd.Series) -> str:
    n = len(target_series.dropna())
    if target_series.dtype == object:
        return "classification"
    nunique = target_series.nunique()
    # Float target with many unique values relative to dataset size → regression
    if target_series.dtype.kind == "f" and nunique > 10 and nunique / max(n, 1) > 0.05:
        return "regression"
    if nunique <= _CLASSIFICATION_UNIQUE_THRESHOLD:
        return "classification"
    return "regression"


def _class_distribution(target_series: pd.Series, problem_type: str) -> dict[str, int]:
    if problem_type != "classification":
        return {}
    return {str(k): int(v) for k, v in target_series.value_counts().items()}


def _describe_feature(series: pd.Series, random_seed: int = 42) -> FeatureInfo:
    missing_pct = round(float(series.isnull().mean() * 100), 2)
    unique_vals = int(series.nunique(dropna=True))
    distribution = _infer_distribution(series, random_seed=random_seed)
    return FeatureInfo(
        name=series.name,
        dtype=str(series.dtype),
        unique_values=unique_vals,
        missing_percentage=missing_pct,
        distribution=distribution,
    )


def _infer_distribution(series: pd.Series, random_seed: int = 42) -> str:
    clean = series.dropna()
    if not pd.api.types.is_numeric_dtype(clean) or len(clean) < 8:
        return "unknown"
    skewness = float(clean.skew())
    sample = clean.sample(min(500, len(clean)), random_state=random_seed)
    _, p_value = scipy_stats.shapiro(sample)
    if p_value > 0.05 and abs(skewness) < 0.5:
        return "normal"
    if abs(skewness) >= 0.5:
        return "skewed"
    value_range = float(clean.max() - clean.min())
    expected_std_uniform = value_range / (2 * np.sqrt(3))
    if abs(float(clean.std()) - expected_std_uniform) / (expected_std_uniform + 1e-9) < 0.15:
        return "uniform"
    return "unknown"


def _recommend_metrics(problem_type: str) -> list[str]:
    if problem_type == "classification":
        return ["accuracy", "f1_score"]
    return ["mse", "rmse", "r2"]


def _imbalance_ratio(class_distribution: dict[str, int]) -> float | None:
    if not class_distribution or len(class_distribution) < 2:
        return None
    counts = list(class_distribution.values())
    return round(max(counts) / max(min(counts), 1), 4)


def _feature_variance_mean(num_df: pd.DataFrame) -> float | None:
    if num_df.empty:
        return None
    return round(float(num_df.var(ddof=0).mean()), 6)


def _class_entropy(class_distribution: dict[str, int]) -> float | None:
    if not class_distribution:
        return None
    counts = np.array(list(class_distribution.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return round(entropy, 6)
