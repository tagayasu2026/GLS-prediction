"""Generate a SHAP summary plot for a saved tree-based PyCaret model.

This script is intended for model interpretation after training. It may require
minor adaptation depending on the final PyCaret pipeline object and model type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from pycaret.classification import load_model

from config import IGNORE_FEATURES, OPTIONAL_MISSING_EXTERNAL_FEATURES, TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument("--data", required=True, help="CSV file for SHAP analysis.")
    parser.add_argument("--model-path", required=True, help="Path stem of saved PyCaret model.")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output", default="results/shap_summary.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    for col in OPTIONAL_MISSING_EXTERNAL_FEATURES:
        if col not in data.columns:
            data[col] = pd.NA

    model = load_model(args.model_path)

    drop_cols = [args.target] + [c for c in IGNORE_FEATURES if c in data.columns]
    x_data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # PyCaret models are pipelines. The final estimator is usually the last step.
    # The preprocessing steps are applied before SHAP calculation.
    if hasattr(model, "steps"):
        preprocessing = model[:-1]
        estimator = model[-1]
        x_processed = preprocessing.transform(x_data)
        feature_names = getattr(preprocessing, "get_feature_names_out", lambda: x_data.columns)()
        x_processed = pd.DataFrame(x_processed, columns=feature_names)
    else:
        estimator = model
        x_processed = x_data

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(x_processed)

    if isinstance(shap_values, list):
        values_to_plot = shap_values[1]
    else:
        values_to_plot = shap_values
        if getattr(values_to_plot, "ndim", 2) == 3:
            values_to_plot = values_to_plot[:, :, 1]

    shap.summary_plot(values_to_plot, x_processed, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved SHAP summary plot to: {output}")


if __name__ == "__main__":
    main()
