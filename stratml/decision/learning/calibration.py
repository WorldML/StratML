"""
calibration.py
--------------
Decision/Learning — Calibration Layer.

Active path: IsotonicRegression fitted on (predicted_gain, actual_gain) pairs
             from decision_dataset.csv when >= 50 rows are available.
Stub path:   Pass-through until enough data exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

from stratml.decision.learning.value_model import ValuePrediction, _MIN_ROWS
import stratml.decision.learning.dataset_builder as _db

log = logging.getLogger(__name__)


def _load_calibration_pairs(csv_path: Path):
    """Return (predicted, actual) arrays, or (None, None) if insufficient data."""
    if not csv_path.exists():
        return None, None
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        df["pred_num"] = pd.to_numeric(df["predicted_gain"], errors="coerce")
        df["obs_num"] = pd.to_numeric(df["observed_gain"], errors="coerce")
        df = df.dropna(subset=["pred_num", "obs_num"])
        if len(df) < _MIN_ROWS:
            return None, None
        y_pred = df["pred_num"].values
        y_true = df["obs_num"].values
        return y_pred, y_true
    except Exception as exc:
        log.warning("calibration: failed to load pairs (%s)", exc)
        return None, None


def calibrate(predictions: list[ValuePrediction]) -> list[ValuePrediction]:
    """Correct predicted gains using fitted IsotonicRegression when data is available."""
    y_pred_train, y_true_train = _load_calibration_pairs(_db._UNIFIED_PATH)

    if y_pred_train is not None:
        try:
            import numpy as np
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_pred_train, y_true_train)

            calibrated = []
            for p in predictions:
                adjusted = float(iso.predict([p.predicted_gain])[0])
                calibrated.append(ValuePrediction(
                    action_type=p.action_type,
                    parameters=p.parameters,
                    predicted_gain=round(max(0.0, min(adjusted, 1.0)), 4),
                    predicted_cost=p.predicted_cost,
                ))
            return calibrated
        except Exception as exc:
            log.warning("calibration: isotonic fit failed (%s), pass-through", exc)

    return predictions
