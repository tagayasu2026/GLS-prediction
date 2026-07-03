"""External validation of saved PyCaret models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import load_model, predict_model

from config import MODEL_NAMES, OPTIONAL_MISSING_EXTERNAL_FEATURES, TARGET
from metrics import classification_metrics, get_positive_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external validation.")
    parser.add_argument("--data", required=True, help="Path to external validation CSV file.")
    parser.add_argument("--model-dir", default="models", help="Directory containing saved models.")
    parser.add_argument("--target", default=TARGET, help="Target column name.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability threshold.")
    parser.add_argument("--output", default="results/external_validation_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    for col in OPTIONAL_MISSING_EXTERNAL_FEATURES:
        if col not in data.columns:
            data[col] = pd.NA

    rows = []
    for public_name, file_stem in MODEL_NAMES.items():
        model_path = model_dir / file_stem
        print(f"Evaluating {public_name} from {model_path}...")
        model = load_model(str(model_path))
        pred = predict_model(model, data=data.copy(), raw_score=True)
        y_true = pred[args.target].astype(int).to_numpy()
        y_prob = get_positive_probability(pred)
        metrics = classification_metrics(y_true, y_prob, threshold=args.threshold)
        metrics = {"model": public_name, **metrics}
        rows.append(metrics)

    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(results)
    print(f"Saved metrics to: {output}")


if __name__ == "__main__":
    main()
