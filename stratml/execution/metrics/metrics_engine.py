"""
metrics_engine.py
-----------------
Phase 6 — Convert raw predictions and probabilities into a validated ExperimentMetrics object.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, log_loss as sk_log_loss,
)

from stratml.execution.schemas import ExperimentMetrics


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    train_curve: list[float],
    val_curve: list[float],
    problem_type: str,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence] = None,
    task_type: Optional[str] = None,
) -> ExperimentMetrics:
    """Compute ExperimentMetrics from predictions, probabilities, and loss curves.

    Supports:
    - Binary classification: ROC-AUC (primary), F1 (secondary), accuracy, precision, recall
    - Multiclass classification: Log Loss (primary), F1 (secondary), accuracy, precision, recall, multiclass ROC-AUC
    - Regression: RMSE (primary), MAE (secondary), R² (secondary), MSE
    - Edge cases: single-class batches, zero-variance outputs, probability clipping
    """
    train_loss = round(float(train_curve[-1]), 6) if train_curve else None
    val_loss   = round(float(val_curve[-1]),   6) if val_curve   else None

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if problem_type == "classification":
        avg = "weighted"
        acc = round(float(accuracy_score(y_true_arr, y_pred_arr)), 6)
        f1  = round(float(f1_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)), 6)
        prec = round(float(precision_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)), 6)
        rec  = round(float(recall_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)), 6)

        roc_auc_val: Optional[float] = None
        log_loss_val: Optional[float] = None

        # Resolve unique classes
        if classes is not None and len(classes) > 0:
            resolved_classes = list(classes)
        else:
            resolved_classes = list(pd.unique(y_true_arr))
            try:
                resolved_classes = sorted(resolved_classes)
            except Exception:
                pass

        n_classes = len(resolved_classes)
        is_binary = (task_type == "binary_classification") or (task_type is None and n_classes <= 2)

        # ── ROC-AUC computation ───────────────────────────────────────────
        if y_proba is not None:
            try:
                y_p = np.asarray(y_proba)
                unique_true = np.unique(y_true_arr)
                if len(unique_true) < 2:
                    # Edge case: single-class batch has undefined ROC-AUC
                    roc_auc_val = 0.5
                elif is_binary:
                    # Binary classification ROC-AUC
                    if y_p.ndim == 2 and y_p.shape[1] >= 2:
                        pos_label = resolved_classes[1] if len(resolved_classes) >= 2 else unique_true[-1]
                        y_true_bin = (y_true_arr == pos_label).astype(int)
                        if len(np.unique(y_true_bin)) > 1:
                            roc_auc_val = round(float(roc_auc_score(y_true_bin, y_p[:, 1])), 6)
                        else:
                            roc_auc_val = 0.5
                    elif y_p.ndim == 1 or (y_p.ndim == 2 and y_p.shape[1] == 1):
                        pos_label = resolved_classes[1] if len(resolved_classes) >= 2 else unique_true[-1]
                        y_true_bin = (y_true_arr == pos_label).astype(int)
                        if len(np.unique(y_true_bin)) > 1:
                            roc_auc_val = round(float(roc_auc_score(y_true_bin, y_p.ravel())), 6)
                        else:
                            roc_auc_val = 0.5
                else:
                    # Multiclass ROC-AUC (OvR macro)
                    if y_p.ndim == 2 and y_p.shape[1] == n_classes:
                        roc_auc_val = round(float(roc_auc_score(
                            y_true_arr, y_p, multi_class="ovr", average="macro", labels=resolved_classes
                        )), 6)
            except Exception:
                roc_auc_val = None

        # ── Log Loss computation ──────────────────────────────────────────
        if y_proba is not None:
            try:
                y_p = np.asarray(y_proba, dtype=float)
                eps = 1e-15
                if y_p.ndim == 1:
                    y_p = np.column_stack([1.0 - y_p, y_p])
                # Clip to prevent log(0)
                y_p = np.clip(y_p, eps, 1.0 - eps)
                # Re-normalize rows to guarantee sum to 1
                row_sums = y_p.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_p = y_p / row_sums

                log_loss_val = round(float(sk_log_loss(y_true_arr, y_p, labels=resolved_classes)), 6)
            except Exception:
                log_loss_val = None

        return ExperimentMetrics(
            accuracy=acc,
            f1_score=f1,
            precision=prec,
            recall=rec,
            roc_auc=roc_auc_val,
            log_loss=log_loss_val,
            train_loss=train_loss,
            validation_loss=val_loss,
        )

    else:
        mse  = round(float(mean_squared_error(y_true_arr, y_pred_arr)), 6)
        rmse = round(float(np.sqrt(max(0.0, mse))), 6)
        mae  = round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 6)
        r2   = round(float(r2_score(y_true_arr, y_pred_arr)), 6)
        return ExperimentMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_loss=train_loss,
            validation_loss=val_loss,
        )
