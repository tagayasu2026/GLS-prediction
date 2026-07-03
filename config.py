"""Configuration settings for GLS prediction model scripts.

This file defines the variable names and model file names used by the
training, external validation, and SHAP analysis scripts.

Modify these lists if your dataset uses different column names.
"""

from __future__ import annotations

# Target variable
# 1 = Low-GLS, 0 = Normal-GLS
TARGET = "GLSC"

# Categorical variables used in the analysis.
# These are converted to categorical dtype before PyCaret setup.
CATEGORICAL_FEATURES = [
    "Sex",
    "AR",
    "MR",
    "TR",
    "PR",
    "Pericardial_effusion",
]

# Columns to ignore during model training and validation.
# Add patient identifiers, dates, raw GLS values, or other non-predictor columns here.
IGNORE_FEATURES = [
    "ID",
    "ID1",
    "ID2",
    "ID3",
    "Date",
    "GLS",
]

# Saved model file names.
# PyCaret's save_model adds the .pkl extension automatically.
MODEL_NAMES = {
    "catboost": "catboost_model",
    "extra_trees": "extra_trees_model",
    "random_forest": "random_forest_model",
    "logistic_regression": "logistic_regression_model",
}

# Optional list of expected predictor variables.
# This is provided as documentation and can be used for column checking.
ECHO_FEATURES = [
    "Age",
    "Sex",
    "BMI",
    "EF",
    "AAD",
    "LAD",
    "LVDd",
    "LVDs",
    "IVST",
    "PWT",
    "E",
    "A",
    "DCT",
    "Septal_e",
    "Septal_a",
    "Lateral_e",
    "Lateral_a",
    "E_A",
    "E_e",
    "LVMI",
    "AV_Vmax",
    "LVOT_Vmax",
    "AR",
    "MR",
    "TR",
    "PR",
    "Pericardial_effusion",
]
