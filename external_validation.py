"""External validation of saved PyCaret models.

This script loads saved PyCaret model pipelines, applies them to an external
validation dataset, calculates performance metrics, and optionally saves ROC
curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pycaret.classification import load_model, predict_model
from sklearn.metrics import auc, roc_curve

from config import CATEGORICAL_FEATURES, MODEL_NAMES, OPTIONAL_MISSING_EXTERNAL_FEATURES, TARGET
from metrics import classification_metrics, get_positive_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external validation.")
    parser.add_argument("--data", required=True, help="Path to external validation CSV file.")
    parser.add_argument("--model-dir", default="models", help="Directory containing saved models.")
    parser.add_argument("--target", default=TARGET, help="Target column name.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability threshold for binary classification.")
    parser.add_argument("--output", default="results/external_validation_metrics.csv", help="Output CSV file for metrics.")
    parser.add_argument("--roc-output", default="results/external_validation_roc.png", help="Output PNG file for ROC curves.")
    parser.add_argument("--no-roc", action="store_true", help="Do not generate ROC curve figure.")
    return parser.parse_args()


def prepare_external_data(path: str) -> pd.DataFrame:
    """Load and prepare the external validation dataset."""
    data = pd.read_csv(path)

    # Add variables that were absent in some external cohorts as missing values.
    for col in OPTIONAL_MISSING_EXTERNAL_FEATURES:
        if col not in data.columns:
            data[col] = pd.NA

    # Match categorical handling in the original PyCaret workflow.
    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    roc_output = Path(args.roc_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    roc_output.parent.mkdir(parents=True, exist_ok=True)

    data = prepare_external_data(args.data)

    rows = []
    roc_items = []

    for public_name, file_stem in MODEL_NAMES.items():
        model_path = model_dir / file_stem
        print(f"Evaluating {public_name} from {model_path}...")
        model = load_model(str(model_path))

        pred = predict_model(model, data=data.copy(), raw_score=True, verbose=False)
        y_true = pred[args.target].astype(int).to_numpy()
        y_prob = get_positive_probability(pred)

        metric_row = classification_metrics(y_true, y_prob, threshold=args.threshold)
        metric_row = {"Model": public_name, **metric_row}
        rows.append(metric_row)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_items.append((public_name, fpr, tpr, auc(fpr, tpr)))

    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(results.round(3))
    print(f"Saved metrics to: {output}")

    if not args.no_roc:
        plt.figure(figsize=(7, 6))
        for name, fpr, tpr, roc_auc in roc_items:
            plt.plot(fpr, tpr, label=f"{name} AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Reference")
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title("External validation ROC curves")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(roc_output, dpi=300, bbox_inches="tight")
        print(f"Saved ROC figure to: {roc_output}")


if __name__ == "__main__":
    main()
