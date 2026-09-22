"""
uncertainty.py
--------------
Decision/Learning — Uncertainty Estimator.

Active path: Ensemble of 5 RandomForestRegressors; variance across predictions
             is the confidence signal. Activated when >= 50 rows exist.
Stub path:   confidence=0.5, variance=0.0 until ensemble is trained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from stratml.core.schemas import StateObject
from stratml.decision.learning.value_model import (
    ValuePrediction,
    _DATASET_PATH,
    _MIN_ROWS,
    _load_training_data,
    _encode_state_action,
)

log = logging.getLogger(__name__)

_N_ESTIMATORS = 5


@dataclass
class UncertaintyEstimate:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float
    confidence: float
    variance: float = 0.0


def estimate(predictions: list[ValuePrediction], state: "StateObject | None" = None) -> list[UncertaintyEstimate]:
    """Wrap each prediction with confidence/variance from ensemble when data is available."""
    X_train, y_train = _load_training_data(_DATASET_PATH)

    if X_train is not None:
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            models = [
                RandomForestRegressor(n_estimators=20, random_state=seed).fit(X_train, y_train)
                for seed in range(_N_ESTIMATORS)
            ]

            results = []
            for p in predictions:
                if state is not None:
                    x = [_encode_state_action(state, p.action_type)]
                else:
                    x = [[p.predicted_gain] + [0.0] * 12]  # proxy fallback
                preds = np.array([m.predict(x)[0] for m in models])
                variance = float(np.var(preds))
                mean_pred = float(np.mean(preds))
                confidence = round(max(0.0, min(1.0 - variance * 10, 1.0)), 4)
                results.append(UncertaintyEstimate(
                    action_type=p.action_type,
                    parameters=p.parameters,
                    predicted_gain=round(max(0.0, min(mean_pred, 1.0)), 4),
                    predicted_cost=p.predicted_cost,
                    confidence=confidence,
                    variance=round(variance, 6),
                ))
            return results
        except Exception as exc:
            log.warning("uncertainty: ensemble failed (%s), using stub", exc)

    # Stub fallback
    return [
        UncertaintyEstimate(
            action_type=p.action_type,
            parameters=p.parameters,
            predicted_gain=p.predicted_gain,
            predicted_cost=p.predicted_cost,
            confidence=0.5,
            variance=0.0,
        )
        for p in predictions
    ]
