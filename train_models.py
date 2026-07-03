"""Train and tune PyCaret classification models for Low-GLS prediction.

This script follows the original PyCaret workflow used for model development.
It does not include patient data or trained model files.

Before running:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import ClassificationExperiment

from config import CATEGORICAL_FEATURES, IGNORE_FEATURES, MODEL_IDS, MODEL_NAMES, TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyCaret GLS prediction models.")
    parser.add_argument("--data", required=True, help="Path to development CSV file.")
    parser.add_argument("--target", default=TARGET, help="Target column name.")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained models.")
    parser.add_argument("--results-dir", default="results", help="Directory to save internal validation results.")
    parser.add_argument("--session-id", type=int, default=1, help="Random seed.")
    parser.add_argument("--train-size", type=float, default=0.8, help="Training fraction used by PyCaret setup.")
    parser.add_argument("--include-lightgbm", action="store_true", help="Also train LightGBM as an optional exploratory model.")
    return parser.parse_args()


def specificity(y_true, y_pred):
    """Specificity metric for PyCaret model comparison."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def npv(y_true, y_pred):
    """Negative predictive value metric for PyCaret model comparison."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    categorical_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for col in categorical_features:
        df[col] = df[col].astype("category")

    ignore_features = [c for c in IGNORE_FEATURES if c in df.columns]

    exp = ClassificationExperiment()
    exp.setup(
        data=df,
        target=args.target,
        train_size=args.train_size,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.6,
        normalize=True,
        normalize_method="zscore",
        categorical_features=categorical_features,
        ignore_features=ignore_features,
        session_id=args.session_id,
    )

    # Add specificity and NPV, as in the original notebook workflow.
    exp.add_metric("Spec.", "Spec.", specificity)
    exp.add_metric("NPV", "NPV", npv)

    model_ids = dict(MODEL_IDS)
    model_names = dict(MODEL_NAMES)
    if args.include_lightgbm:
        model_ids["lightgbm"] = "lightgbm"
        model_names["lightgbm"] = "tuned_lightgbm"

    internal_rows = []

    for public_name, pycaret_id in model_ids.items():
        print(f"Creating {public_name} model ({pycaret_id})...")
        model = exp.create_model(pycaret_id)

        print(f"Tuning {public_name} model using Optuna and AUC optimization...")
        tuned_model = exp.tune_model(
            model,
            search_library="optuna",
            optimize="AUC",
            choose_better=False,
        )

        output_path = model_dir / model_names[public_name]
        exp.save_model(tuned_model, str(output_path))
        print(f"Saved: {output_path}.pkl")

        # PyCaret predict_model without data evaluates the internal hold-out set.
        internal_result = exp.predict_model(tuned_model, raw_score=True, verbose=False)
        metrics_row = exp.pull().iloc[0].to_dict()
        metrics_row["Model"] = public_name
        internal_rows.append(metrics_row)

    internal_results = pd.DataFrame(internal_rows)
    internal_output = results_dir / "internal_validation_metrics.csv"
    internal_results.to_csv(internal_output, index=False)
    print(f"Internal validation metrics saved to: {internal_output}")
    print(f"Models saved to: {model_dir}")


if __name__ == "__main__":
    main()
