"""Utility functions for model performance evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN if denominator is zero."""
    return numerator / denominator if denominator != 0 else np.nan


def get_positive_probability(prediction_df: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probabilities from PyCaret prediction output.

    PyCaret column names can differ slightly across versions. This helper
    checks the common column names used by PyCaret 3.x.
    """
    candidate_columns = [
        "prediction_score_1",
        "Score_1",
        "probability_1",
        "prediction_score",
        "Score",
    ]

    for col in candidate_columns:
        if col in prediction_df.columns:
            return prediction_df[col].astype(float).to_numpy()

    score_cols = [c for c in prediction_df.columns if c.startswith("prediction_score")]
    if score_cols:
        return prediction_df[sorted(score_cols)[-1]].astype(float).to_numpy()

    raise ValueError(
        "Could not identify a positive-class probability column in PyCaret output. "
        f"Available columns: {list(prediction_df.columns)}"
    )


def classification_metrics(
    y_true,
    y_prob,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Calculate AUC, accuracy, sensitivity, specificity, PPV, NPV, and F1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Threshold": threshold,
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": safe_divide(tp, tp + fn),
        "Specificity": safe_divide(tn, tn + fp),
        "PPV": safe_divide(tp, tp + fp),
        "NPV": safe_divide(tn, tn + fn),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
