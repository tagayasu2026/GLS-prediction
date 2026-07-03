"""Train and tune PyCaret classification models for Low-GLS prediction.

This script does not include patient data or trained model files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import ClassificationExperiment

from config import CATEGORICAL_FEATURES, IGNORE_FEATURES, MODEL_NAMES, TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyCaret GLS prediction models.")
    parser.add_argument("--data", required=True, help="Path to development CSV file.")
    parser.add_argument("--target", default=TARGET, help="Target column name.")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained models.")
    parser.add_argument("--session-id", type=int, default=1, help="Random seed.")
    parser.add_argument("--train-size", type=float, default=0.8, help="Training fraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

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

    model_ids = {
        "catboost": "catboost",
        "extra_trees": "et",
        "random_forest": "rf",
        "logistic_regression": "lr",
    }

    for public_name, pycaret_id in model_ids.items():
        print(f"Training {public_name}...")
        model = exp.create_model(pycaret_id)
        tuned_model = exp.tune_model(
            model,
            search_library="optuna",
            optimize="AUC",
            choose_better=False,
        )
        output_path = model_dir / MODEL_NAMES[public_name]
        exp.save_model(tuned_model, str(output_path))

    print(f"Models saved to: {model_dir}")


if __name__ == "__main__":
    main()
