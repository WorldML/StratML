"""
loader.py
---------
Phase 1 — Dataset Ingestion: load a dataset file into a pandas DataFrame.

Supported formats:
    .csv   — comma-separated
    .tsv   — tab-separated
    .json  — records or columns orientation
    .parquet
    .xlsx / .xls — Excel (requires openpyxl)
    .arff  — Weka/OpenML ARFF files (requires scipy)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def _load_arff(p: Path) -> pd.DataFrame:
    """Load ARFF file decoding byte strings to UTF-8 and converting missing values."""
    import scipy.io.arff as arff
    data, meta = arff.loadarff(str(p))
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].dropna()
            if not s.empty and isinstance(s.iloc[0], bytes):
                df[col] = df[col].str.decode("utf-8", errors="replace")
            # Convert '?' missing indicator strings to np.nan
            df[col] = df[col].map(lambda x: np.nan if (isinstance(x, str) and x.strip() == "?") else x)
    return df


_LOADERS = {
    ".csv":     lambda p: pd.read_csv(p),
    ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
    ".json":    lambda p: pd.read_json(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx":    lambda p: pd.read_excel(p),
    ".xls":     lambda p: pd.read_excel(p),
    ".arff":    _load_arff,
}


def load_dataframe(path: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Load a dataset file into a DataFrame.

    Returns:
        (dataframe, dataset_name) where dataset_name is the file stem.

    Raises:
        FileNotFoundError: file does not exist.
        ValueError: unsupported file format.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {list(_LOADERS)}")

    df = loader(p)
    return df, p.stem
