"""Configuration for GLS prediction model scripts.

Edit this file if your dataset uses different column names.
"""

from __future__ import annotations

# Target variable: 1 = Low-GLS, 0 = Normal-GLS
TARGET = "GLSC"

# Variables that should not be used as predictors.
IGNORE_FEATURES = [
    "ID",
    "ID1",
    "ID2",
    "ID3",
    "Date",
    "GLS",
]

# Categorical variables used in the original PyCaret workflow.
CATEGORICAL_FEATURES = [
    "Gender",
    "MR",
    "Peri",
    "TR",
    "PR",
    "AR",
]

# Original predictor variables: demographics + 25 conventional echo parameters.
# The external validation script adds unavailable variables as missing values
# when they are absent in the external cohort.
PREDICTOR_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "AR",
    "AAD",
    "AV_peak",
    "A",
    "DCT",
    "EF",
    "IVST",
    "LAD",
    "LVDd",
    "LVDs",
    "LVMI",
    "LVOT_Vmax",
    "PWT",
    "MR",
    "Peri",
    "PR",
    "Septal_a",
    "Lateral_a",
    "Septal_e",
    "Lateral_e",
    "E_A",
    "E_e",
    "E",
    "TR",
]

# Variables unavailable in some external validation cohorts.
OPTIONAL_MISSING_EXTERNAL_FEATURES = [
    "AV_peak",
    "AAD",
    "Peri",
    "LVMI",
]

# Public model names and corresponding PyCaret model identifiers.
# These reflect the models described in the revised manuscript.
MODEL_IDS = {
    "catboost": "catboost",
    "extra_trees": "et",
    "random_forest": "rf",
    "logistic_regression": "lr",
}

# Saved model file stems. PyCaret adds the .pkl extension automatically.
MODEL_NAMES = {
    "catboost": "tuned_catboost",
    "extra_trees": "tuned_extra_trees",
    "random_forest": "tuned_random_forest",
    "logistic_regression": "tuned_logistic_regression",
}
