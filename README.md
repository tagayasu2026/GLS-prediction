# External Validation of Machine Learning Models Predicting Global Longitudinal Strain from Conventional Echocardiography in Patients with Cancer

This repository contains Python source code used to develop, externally validate, and interpret machine learning models described in the manuscript:

**Anzai T, et al.**  
*External Validation of Machine Learning Models Predicting Global Longitudinal Strain from Conventional Echocardiography in Patients with Cancer.*

The code is provided to support reproducibility and future external validation. Patient-level data and trained model files are **not** included.

## Repository contents

- `train_models.py`  
  Trains and tunes PyCaret classification models using the development cohort. The script follows the original PyCaret workflow and trains CatBoost, Extra Trees, Random Forest, and Logistic Regression.

- `external_validation.py`  
  Loads saved PyCaret model pipelines and evaluates them in an external validation cohort. The script outputs AUC, accuracy, sensitivity, specificity, PPV, NPV, F1 score, and ROC curves.

- `shap_analysis.py`  
  Generates SHAP summary plots for a saved tree-based PyCaret model, especially the CatBoost model used for model interpretation.

- `config.py`  
  Defines target variable, categorical variables, ignored variables, unavailable external variables, and model file names.

- `metrics.py`  
  Provides utility functions for classification metrics and probability extraction from PyCaret output.

## Data availability

The original clinical datasets cannot be publicly shared because they contain protected patient information and are subject to institutional ethics restrictions.

Users may run these scripts using their own datasets with the same variable names and structure as described in the manuscript.

The trained models are not included because they were developed using patient data subject to institutional ethics restrictions.

## Installation

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.11
PyCaret 3.2.0
scikit-learn 1.2.2
joblib 1.3.2
```

## Usage

### 1. Train and tune models

```bash
python train_models.py \
  --data development_dataset.csv \
  --target GLSC \
  --model-dir models \
  --results-dir results \
  --session-id 1 \
  --train-size 0.8
```

This script saves trained PyCaret pipelines to `models/` and internal validation metrics to `results/internal_validation_metrics.csv`.

### 2. Run external validation

```bash
python external_validation.py \
  --data external_validation_dataset.csv \
  --model-dir models \
  --target GLSC \
  --threshold 0.50 \
  --output results/external_validation_metrics.csv \
  --roc-output results/external_validation_roc.png
```

For a sensitivity-prioritized threshold analysis, change the threshold value:

```bash
python external_validation.py \
  --data external_validation_dataset.csv \
  --model-dir models \
  --threshold 0.25 \
  --output results/external_validation_metrics_threshold_025.csv
```

### 3. Run SHAP analysis

```bash
python shap_analysis.py \
  --data external_validation_dataset.csv \
  --model-path models/tuned_catboost \
  --target GLSC \
  --output results/shap_summary.png \
  --max-display 20
```

Note: `--model-path` should be the saved PyCaret model path stem, usually without the `.pkl` extension.

## Notes

- The scripts do not contain patient-level data.
- The scripts do not include trained model files.
- Model performance from the manuscript cannot be reproduced without the original clinical datasets.
- If your dataset uses different column names, edit `config.py` before running the scripts.

## Citation

If you use this code, please cite the corresponding manuscript.
