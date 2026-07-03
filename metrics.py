"""Utility functions for model performance evaluation.

The functions in this file are used by the external validation script to
calculate standard binary classification metrics for Low-GLS prediction.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Calculate binary classification metrics.

    Parameters
    ----------
    y_true:
        True binary labels. Expected values are 0 and 1.
    y_prob:
        Predicted probabilities for the positive class.
    threshold:
        Probability threshold used to convert probabilities into binary labels.

    Returns
    -------
    dict
        Dictionary containing AUC, accuracy, sensitivity, specificity, PPV,
        NPV, and F1 score.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return {
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Threshold": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def metrics_to_dataframe(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert a nested metrics dictionary to a tidy DataFrame.

    Parameters
    ----------
    metrics:
        Dictionary in the form {"model_name": {"AUC": ..., "Accuracy": ...}}.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per model.
    """

    rows = []
    for model_name, values in metrics.items():
        row = {"Model": model_name}
        row.update(values)
        rows.append(row)

    return pd.DataFrame(rows)


def get_positive_class_probability(prediction_df: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probabilities from PyCaret prediction output.

    PyCaret output column names can vary depending on version. This helper
    attempts common probability column names.

    Parameters
    ----------
    prediction_df:
        Output DataFrame returned by PyCaret's predict_model.

    Returns
    -------
    numpy.ndarray
        Predicted probabilities for the positive class.
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
            return prediction_df[col].to_numpy(dtype=float)

    raise ValueError(
        "Could not find a positive-class probability column in PyCaret output. "
        f"Available columns: {list(prediction_df.columns)}"
    )
