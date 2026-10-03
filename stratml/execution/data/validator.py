"""
validator.py
------------
Phase 1 — Dataset Ingestion: validate the loaded DataFrame and build a Dataset object.
"""

from typing import Optional
import hashlib
import warnings

import pandas as pd
from stratml.execution.schemas import Dataset



def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a deterministic 16-hex-character content fingerprint of a pandas DataFrame.
    Same column names, dtypes, and values produce the identical fingerprint.
    Any change in values, row order, rows added/removed, or column names alters the fingerprint.
    """
    hasher = hashlib.sha256()
    hasher.update(str(list(df.columns)).encode("utf-8"))
    hasher.update(str([str(dt) for dt in df.dtypes]).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
        hasher.update(row_hashes.tobytes())
    except Exception:
        for col in df.columns:
            hasher.update(str(df[col].tolist()).encode("utf-8", errors="replace"))
    return hasher.hexdigest()[:16]


BENCHMARK_TARGETS: dict[str, str] = {
    "adult": "class",
    "australian": "A15",
    "bank-marketing": "Class",
    "blood-transfusion": "Class",
    "credit-g": "class",
    "jannis": "class",
    "jasmine": "class",
    "kc1": "defects",
    "phoneme": "Class",
    "vehicle": "Class",
    "abalone": "Class_number_of_rings",
    "boston": "MEDV",
    "brazilian": "total_(BRL)",
    "elevators": "Goal",
    "house_16H": "price",
    "moneyball": "RS",
    "online-news": "shares",
    "pol": "foo",
    "wine_quality": "quality",
    "yprop_4_1": "oz252",
}


def build_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str] = None,
) -> Dataset:
    """
    Validate the DataFrame and construct a Dataset object.

    Raises:
        ValueError: on structural problems that would break downstream steps.
    """
    # 1. Zero rows
    if len(df) == 0:
        raise ValueError("Dataset has 0 rows.")

    # 2. Duplicate column names
    seen, dupes = set(), set()
    for c in df.columns:
        (dupes if c in seen else seen).add(c)
    if dupes:
        raise ValueError(f"Duplicate column names detected: {sorted(dupes)}")

    # 3. Resolve target column if None or not found
    if target_column is None or target_column == "":
        name_key = dataset_name.lower().replace("_", "-")
        if name_key in BENCHMARK_TARGETS and BENCHMARK_TARGETS[name_key] in df.columns:
            target_column = BENCHMARK_TARGETS[name_key]
        elif dataset_name in BENCHMARK_TARGETS and BENCHMARK_TARGETS[dataset_name] in df.columns:
            target_column = BENCHMARK_TARGETS[dataset_name]
        else:
            for cand in ["class", "Class", "target", "Target", "label", "Label"]:
                if cand in df.columns:
                    target_column = cand
                    break
            if target_column is None:
                target_column = df.columns[-1]

    if target_column not in df.columns:
        # Check case-insensitive match
        col_map = {c.lower(): c for c in df.columns}
        if target_column.lower() in col_map:
            target_column = col_map[target_column.lower()]
        else:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )


    # 4. All-null columns — warn and drop
    all_null = [c for c in df.columns if df[c].isnull().all()]
    if all_null:
        warnings.warn(f"Dropping all-null columns: {all_null}", UserWarning, stacklevel=2)
        df = df.drop(columns=all_null)

    # 5. Target validity
    target = df[target_column]
    if target.isnull().all():
        raise ValueError(f"Target column '{target_column}' is entirely null.")
    if target.dropna().nunique() < 2:
        raise ValueError(
            f"Target column '{target_column}' has fewer than 2 unique non-null values — "
            "not a valid ML problem."
        )

    fingerprint = compute_dataset_fingerprint(df)

    return Dataset(
        dataset_name=dataset_name,
        rows=len(df),
        columns=len(df.columns),
        target_column=target_column,
        dataset_type="tabular",
        raw_dataframe=df,
        dataset_fingerprint=fingerprint,
    )
