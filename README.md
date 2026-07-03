# External Validation of Machine Learning Models Predicting Global Longitudinal Strain

This repository provides public source code for training and externally validating machine learning models that predict reduced global longitudinal strain (Low-GLS) from conventional echocardiographic parameters in patients with cancer.

The code is provided to support reproducibility and future external validation. Patient-level data and trained model files are **not** included.

## Overview

The workflow includes:

1. Training and internal validation using PyCaret.
2. Hyperparameter tuning of Random Forest, Extra Trees, CatBoost, and Logistic Regression.
3. External validation using saved PyCaret model pipelines.
4. Performance evaluation using AUC, accuracy, sensitivity, specificity, PPV, NPV, and F1 score.
5. Optional threshold sensitivity analysis.
6. Optional SHAP analysis for model interpretation.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── example_data/
│   ├── sample_development_data.csv
│   └── sample_external_data.csv
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── src/
    ├── config.py
    ├── metrics.py
    ├── train_models.py
    ├── external_validation.py
    ├── threshold_analysis.py
    ├── shap_analysis.py
    └── make_sample_data.py
```

## Data format

The expected target column is:

- `GLSC`: binary outcome, where 1 indicates Low-GLS and 0 indicates Normal-GLS.

Columns that are not used as predictors can be included but will be ignored if listed in `IGNORE_FEATURES` in `src/config.py`.

The default predictor set is configured in `src/config.py`. Please modify this file if your dataset uses different variable names.

## Installation

```bash
pip install -r requirements.txt
```

The original analysis used PyCaret 3.2.0. For best reproducibility, use the package versions listed in `requirements.txt`.

## Example usage

Create example data:

```bash
python src/make_sample_data.py
```

Train models using the example development dataset:

```bash
python src/train_models.py \
  --data example_data/sample_development_data.csv \
  --target GLSC \
  --model-dir models
```

Run external validation:

```bash
python src/external_validation.py \
  --data example_data/sample_external_data.csv \
  --model-dir models \
  --output results/external_validation_metrics.csv
```

Run threshold analysis:

```bash
python src/threshold_analysis.py \
  --internal-data example_data/sample_development_data.csv \
  --external-data example_data/sample_external_data.csv \
  --model-path models/tuned_catboost \
  --output results/threshold_analysis.csv
```

Run SHAP analysis:

```bash
python src/shap_analysis.py \
  --data example_data/sample_external_data.csv \
  --model-path models/tuned_catboost \
  --output results/shap_summary.png
```

## Notes

- The example datasets are synthetic and are included only to demonstrate the required data structure.
- Actual patient data cannot be shared because of privacy and institutional restrictions.
- Trained model files are not included in this public repository.
- Model performance from the paper cannot be reproduced using the synthetic example data.

## Citation

If you use this code, please cite the corresponding manuscript.
