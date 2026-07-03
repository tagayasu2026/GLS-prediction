"""Generate a SHAP summary plot for a saved tree-based PyCaret model.

This script is intended for interpretation of tree-based models, especially the
CatBoost model used for SHAP analysis in the manuscript. It applies the
preprocessing steps stored in the saved PyCaret pipeline before calculating SHAP
values for the final estimator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pycaret.classification import load_model

from config import CATEGORICAL_FEATURES, IGNORE_FEATURES, OPTIONAL_MISSING_EXTERNAL_FEATURES, TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis for a saved PyCaret model.")
    parser.add_argument("--data", required=True, help="CSV file for SHAP analysis.")
    parser.add_argument("--model-path", required=True, help="Path stem of the saved PyCaret model, without .pkl.")
    parser.add_argument("--target", default=TARGET, help="Target column name.")
    parser.add_argument("--output", default="results/shap_summary.png", help="Output PNG file for the SHAP summary plot.")
    parser.add_argument("--max-display", type=int, default=20, help="Maximum number of features shown in the SHAP plot.")
    return parser.parse_args()


def load_feature_data(path: str, target: str) -> pd.DataFrame:
    """Load data and remove target/non-predictor columns."""
    data = pd.read_csv(path)

    for col in OPTIONAL_MISSING_EXTERNAL_FEATURES:
        if col not in data.columns:
            data[col] = pd.NA

    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            data[col] = data[col].astype("category")

    drop_cols = [target] + [c for c in IGNORE_FEATURES if c in data.columns]
    return data.drop(columns=[c for c in drop_cols if c in data.columns])


def split_pycaret_pipeline(model):
    """Return preprocessing pipeline and final estimator from a PyCaret model."""
    if hasattr(model, "steps"):
        return model[:-1], model[-1]
    return None, model


def get_feature_names(preprocessing, x_data: pd.DataFrame, x_processed) -> list[str]:
    """Get feature names after preprocessing if available."""
    if preprocessing is not None:
        try:
            names = preprocessing.get_feature_names_out()
            return [str(x) for x in names]
        except Exception:
            pass

    if hasattr(x_processed, "shape") and x_processed.shape[1] == x_data.shape[1]:
        return list(x_data.columns)

    return [f"feature_{i}" for i in range(x_processed.shape[1])]


def select_positive_class_shap_values(shap_values):
    """Select SHAP values for the positive class when needed."""
    if isinstance(shap_values, list):
        return shap_values[1]

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    x_data = load_feature_data(args.data, args.target)
    model = load_model(args.model_path)
    preprocessing, estimator = split_pycaret_pipeline(model)

    if preprocessing is not None:
        x_processed = preprocessing.transform(x_data)
    else:
        x_processed = x_data

    if not isinstance(x_processed, pd.DataFrame):
        feature_names = get_feature_names(preprocessing, x_data, x_processed)
        x_processed = pd.DataFrame(x_processed, columns=feature_names, index=x_data.index)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(x_processed)
    values_to_plot = select_positive_class_shap_values(shap_values)

    shap.summary_plot(
        values_to_plot,
        x_processed,
        max_display=args.max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved SHAP summary plot to: {output}")


if __name__ == "__main__":
    main()
