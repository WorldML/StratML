"""
exporter.py
-----------
Deterministic export utilities for OpenML ARFF datasets to CSV.
Leaves original ARFF benchmark files authoritative and unmodified.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from stratml.execution.data.loader import load_dataframe


def export_arff_to_csv(
    arff_path: str | Path,
    output_dir: str | Path = "data/research/csv",
) -> Path:
    """
    Deterministically export a single ARFF file to CSV with identical decoded values.

    Args:
        arff_path: Path to the input .arff file.
        output_dir: Directory where the output .csv will be written.

    Returns:
        Path to the exported CSV file.
    """
    src = Path(arff_path)
    df, name = load_dataframe(src)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{name}.csv"

    df.to_csv(dst, index=False)
    return dst


def export_all_benchmark_arffs(
    research_dir: str | Path = "data/research",
    output_dir: str | Path = "data/research/csv",
) -> dict[str, Path]:
    """
    Export all 20 benchmark ARFF files to CSV.
    """
    res_path = Path(research_dir)
    out_path = Path(output_dir)
    exported = {}

    for sub in ["classification", "regression"]:
        folder = res_path / sub
        if not folder.exists():
            continue
        sub_out = out_path / sub
        sub_out.mkdir(parents=True, exist_ok=True)
        for arff_file in sorted(folder.glob("*.arff")):
            dst = export_arff_to_csv(arff_file, sub_out)
            exported[arff_file.stem] = dst

    return exported
