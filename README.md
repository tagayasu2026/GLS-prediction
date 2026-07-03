# External Validation of Machine Learning Models Predicting Global Longitudinal Strain from Conventional Echocardiography in Patients with Cancer

This repository contains the Python notebooks used in the following
manuscript:

**Anzai T, et al.**\
*External Validation of Machine Learning Models Predicting Global
Longitudinal Strain from Conventional Echocardiography in Patients with
Cancer.*

## Repository Contents

### 01_train_models.ipynb

-   Data preprocessing
-   Model development using PyCaret
-   Internal validation
-   Selection of the three best-performing models:
    -   Random Forest
    -   Extra Trees
    -   CatBoost
-   Logistic regression as a reference model

### 02_external_validation.ipynb

-   External validation using an independent cohort
-   Performance evaluation
-   ROC analysis
-   Calculation of AUC, sensitivity, specificity, PPV, NPV, accuracy,
    and F1 score

### 03_shap_analysis.ipynb

-   SHAP analysis of the CatBoost model
-   Feature importance
-   SHAP summary plots

## Data Availability

The original clinical datasets are not publicly available because they
contain protected patient information and are subject to institutional
ethics restrictions.

Users may run the notebooks using their own datasets with the same
variable names and structure.

## Requirements

Python 3.11

Major packages: - pycaret - scikit-learn - pandas - numpy - matplotlib -
shap - optuna

See `requirements.txt` for details.

## Notes

This repository provides the source code used in the study.

The trained models and original clinical datasets are not included
because they were developed using patient data subject to institutional
ethics restrictions.

## Citation

If you use this code, please cite:

Anzai T, et al.\
*External Validation of Machine Learning Models Predicting Global
Longitudinal Strain from Conventional Echocardiography in Patients with
Cancer.*
