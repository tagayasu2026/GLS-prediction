# External Validation of Machine Learning Models Predicting Global Longitudinal Strain from Conventional Echocardiography in Patients with Cancer

This repository contains the Python source code used to develop, externally validate, and interpret the machine learning models described in the following manuscript:

**Anzai T, et al.**  
*External Validation of Machine Learning Models Predicting Global Longitudinal Strain from Conventional Echocardiography in Patients with Cancer.*

---

## Repository Contents

### `train_models.py`
Trains and tunes PyCaret classification models for Low-GLS prediction.

Main functions:
- Loads a development CSV dataset
- Performs preprocessing using the PyCaret framework
- Trains and tunes the following models:
  - Random Forest
  - Extra Trees
  - CatBoost
  - Logistic Regression
- Saves trained models to the specified model directory

### `external_validation.py`
Performs external validation of saved PyCaret models.

Main functions:
- Loads an external validation CSV dataset
- Loads saved trained models
- Applies the models to the external validation cohort
- Calculates performance metrics:
  - AUC
  - Accuracy
  - Sensitivity
  - Specificity
  - Positive Predictive Value (PPV)
  - Negative Predictive Value (NPV)
  - F1 score
- Saves performance results as a CSV file

### `shap_analysis.py`
Generates a SHAP summary plot for a saved tree-based PyCaret model.

Main functions:
- Loads a CSV dataset for SHAP analysis
- Loads a saved PyCaret model
- Applies preprocessing steps from the PyCaret pipeline
- Calculates SHAP values
- Saves a SHAP summary plot as a PNG file

---

## Data Availability

The original clinical datasets cannot be publicly shared because they contain protected patient information and are subject to institutional ethics restrictions.

Users may execute the scripts using their own datasets with the same variable names and data structure as described in the manuscript.

The trained models and original clinical datasets are **not included** because they were developed using patient data subject to institutional ethics restrictions.

---

## Installation

Install the required Python packages before running the scripts:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.11
```

Major packages:
- pycaret
- scikit-learn
- pandas
- numpy
- matplotlib
- shap
- optuna
- catboost

---

## Usage

Because the original clinical datasets are not publicly available, users should prepare their own datasets with the same variable names and data structure as described in the manuscript.

### 1. Train the machine learning models

```bash
python train_models.py \
    --data development_dataset.csv \
    --target GLSC \
    --model-dir models \
    --session-id 1 \
    --train-size 0.8
```

Arguments:

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--data` | Yes | None | Path to the development CSV file |
| `--target` | No | Value defined in `config.py` | Target column name |
| `--model-dir` | No | `models` | Directory to save trained models |
| `--session-id` | No | `1` | Random seed |
| `--train-size` | No | `0.8` | Training fraction |

Example:

```bash
python train_models.py --data development_dataset.csv
```

---

### 2. Perform external validation

```bash
python external_validation.py \
    --data external_validation_dataset.csv \
    --model-dir models \
    --target GLSC \
    --threshold 0.50 \
    --output results/external_validation_metrics.csv
```

Arguments:

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--data` | Yes | None | Path to the external validation CSV file |
| `--model-dir` | No | `models` | Directory containing saved trained models |
| `--target` | No | Value defined in `config.py` | Target column name |
| `--threshold` | No | `0.50` | Probability threshold for binary classification |
| `--output` | No | `results/external_validation_metrics.csv` | Output CSV file for performance metrics |

Example:

```bash
python external_validation.py --data external_validation_dataset.csv --model-dir models
```

For sensitivity-prioritized analysis, the threshold can be changed:

```bash
python external_validation.py \
    --data external_validation_dataset.csv \
    --model-dir models \
    --threshold 0.25 \
    --output results/external_validation_metrics_threshold_025.csv
```

---

### 3. Perform SHAP analysis

```bash
python shap_analysis.py \
    --data external_validation_dataset.csv \
    --model-path models/catboost_model \
    --target GLSC \
    --output results/shap_summary.png
```

Arguments:

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--data` | Yes | None | CSV file for SHAP analysis |
| `--model-path` | Yes | None | Path stem of the saved PyCaret model |
| `--target` | No | Value defined in `config.py` | Target column name |
| `--output` | No | `results/shap_summary.png` | Output PNG file for the SHAP summary plot |

Example:

```bash
python shap_analysis.py \
    --data external_validation_dataset.csv \
    --model-path models/catboost_model
```

Note: `--model-path` should be the PyCaret model path stem, not necessarily including the `.pkl` extension.

---

## Important Notes

This repository provides the source code used in the study.

The current scripts depend on shared configuration and utility definitions, including:
- `config.py`
- `metrics.py`

If these files are not included in the repository, users should add them or define the corresponding constants and utility functions before running the scripts.

---

## Citation

If you use this code, please cite:

**Anzai T, et al.**  
*External Validation of Machine Learning Models Predicting Global Longitudinal Strain from Conventional Echocardiography in Patients with Cancer.*
