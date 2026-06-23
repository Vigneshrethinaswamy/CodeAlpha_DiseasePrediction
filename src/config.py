"""
config.py
=========
Centralized configuration for the CodeAlpha_DiseasePrediction project.

This module defines all file paths, hyperparameters, and constants used
throughout the pipeline. Keeping configuration centralized ensures that
every module (data_loader, preprocess, eda, model, train, evaluate,
predict) stays in sync and avoids hardcoded "magic values" scattered
across the codebase.

All paths are resolved relative to the project root, which is computed
dynamically so the project runs correctly regardless of the current
working directory from which it is invoked.
"""

import os

# ---------------------------------------------------------------------------
# Project root and directory structure
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the directory that contains this src/ package's
# parent folder, i.e. CodeAlpha_DiseasePrediction/, regardless of where the
# script is executed from.
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR: str = os.path.join(DATA_DIR, "external")

MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, "plots")
NOTEBOOKS_DIR: str = os.path.join(PROJECT_ROOT, "notebooks")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
RAW_DATA_FILENAME: str = "Diseases_and_Symptoms_dataset.csv"
RAW_DATA_PATH: str = os.path.join(RAW_DATA_DIR, RAW_DATA_FILENAME)

# Processed artifacts (cached splits) to avoid recomputation across runs.
PROCESSED_TRAIN_PATH: str = os.path.join(PROCESSED_DATA_DIR, "train.csv")
PROCESSED_TEST_PATH: str = os.path.join(PROCESSED_DATA_DIR, "test.csv")
FEATURE_COLUMNS_PATH: str = os.path.join(PROCESSED_DATA_DIR, "feature_columns.json")

TARGET_COLUMN: str = "diseases"

# Expected dataset dimensions (used for validation/sanity checks only;
# the pipeline does not hard-fail if these differ slightly, it warns).
EXPECTED_N_SAMPLES: int = 96_088
EXPECTED_N_FEATURES: int = 230
EXPECTED_N_CLASSES: int = 100

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Train/test split configuration
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Model artifact paths
# ---------------------------------------------------------------------------
BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "best_model.pkl")
LABEL_ENCODER_PATH: str = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_PATH: str = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_METADATA_PATH: str = os.path.join(MODELS_DIR, "model_metadata.json")

LOGISTIC_REGRESSION_PATH: str = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RANDOM_FOREST_PATH: str = os.path.join(MODELS_DIR, "random_forest.pkl")
XGBOOST_PATH: str = os.path.join(MODELS_DIR, "xgboost_model.pkl")

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
# NOTE on Logistic Regression:
# The `multi_class` parameter was deprecated in scikit-learn 1.5 and
# REMOVED in scikit-learn >= 1.7 (raises TypeError if passed). Modern
# scikit-learn automatically uses multinomial loss for multi-class problems
# whenever a solver that supports it (e.g. "lbfgs") is selected, so the
# parameter is intentionally omitted here. Do NOT add `multi_class=...`
# back in -- it will break the pipeline on current scikit-learn versions.
LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}

# ---------------------------------------------------------------------------
# EDA configuration
# ---------------------------------------------------------------------------
TOP_N_DISEASES: int = 20
TOP_N_SYMPTOMS: int = 30
CORRELATION_HEATMAP_MAX_FEATURES: int = 40  # subsample of features for readability

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"
LOG_FILE_PATH: str = os.path.join(LOGS_DIR, "pipeline.log")


def ensure_directories() -> None:
    """
    Create all required project directories if they do not already exist.

    This is called at the start of the pipeline (main.py) to guarantee that
    downstream modules (eda.py, train.py, evaluate.py) can write outputs
    without needing to repeat directory-creation logic everywhere.
    """
    required_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        NOTEBOOKS_DIR,
        LOGS_DIR,
    ]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
