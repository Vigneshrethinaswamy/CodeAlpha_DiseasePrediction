"""
data_loader.py
===============
Handles loading and structural validation of the disease-symptom dataset.

Responsibilities:
    - Load the raw CSV from disk.
    - Validate that the target column and expected feature structure exist.
    - Identify symptom (feature) columns programmatically rather than
      hardcoding all 230 names, so the module remains robust to minor
      schema variations in the source CSV.
    - Provide a single source of truth for "what is a feature column"
      that is reused by preprocess.py, eda.py, and predict.py.
"""

import os
from typing import List, Tuple

import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_dataset(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw disease-symptom dataset from a CSV file.

    Args:
        path: Path to the CSV file. Defaults to `config.RAW_DATA_PATH`.

    Returns:
        A pandas DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If no file exists at `path`.
        ValueError: If the file is empty or cannot be parsed as CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please place "
            f"'{config.RAW_DATA_FILENAME}' inside '{config.RAW_DATA_DIR}'."
        )

    logger.info("Loading raw dataset from: %s", path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset at '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset at '{path}' could not be parsed as valid CSV.") from exc

    if df.empty:
        raise ValueError(f"Dataset at '{path}' loaded but contains zero rows.")

    logger.info("Loaded dataset with shape: %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame, target_column: str = config.TARGET_COLUMN) -> List[str]:
    """
    Identify the symptom (feature) columns in the dataset.

    All columns except the target column are treated as symptom features.
    This is more robust than hardcoding 230 column names, since it
    automatically adapts if the dataset schema changes slightly.

    Args:
        df: The loaded dataset.
        target_column: Name of the target column to exclude.

    Returns:
        A list of feature column names, in their original order.

    Raises:
        ValueError: If the target column is not present in `df`.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns. "
            f"Available columns (first 10): {list(df.columns)[:10]}"
        )
    feature_columns = [col for col in df.columns if col != target_column]
    return feature_columns


def validate_dataset(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    expected_n_samples: int = config.EXPECTED_N_SAMPLES,
    expected_n_features: int = config.EXPECTED_N_FEATURES,
    expected_n_classes: int = config.EXPECTED_N_CLASSES,
) -> Tuple[List[str], int]:
    """
    Validate the structural integrity of the loaded dataset.

    Checks performed:
        1. Target column exists.
        2. No missing values anywhere in the dataset.
        3. All feature columns are binary (0/1) valued.
        4. Logs warnings (does not raise) if sample count, feature count,
           or class count differ from the documented dataset properties,
           since minor version-to-version dataset variations should not
           crash the pipeline.

    Args:
        df: The loaded dataset.
        target_column: Name of the target column.
        expected_n_samples: Expected number of rows (for sanity-check warning only).
        expected_n_features: Expected number of symptom feature columns (warning only).
        expected_n_classes: Expected number of unique disease classes (warning only).

    Returns:
        A tuple of (feature_columns, n_classes).

    Raises:
        ValueError: If the target column is missing, if there are missing
            values, or if any feature column is not binary.
    """
    feature_columns = get_feature_columns(df, target_column)

    # --- Missing values check -------------------------------------------------
    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        raise ValueError(
            f"Dataset contains {n_missing} missing values, but the documented "
            f"dataset property states 'No missing values'. Please inspect the source CSV."
        )

    # --- Binary feature check --------------------------------------------------
    non_binary_columns = []
    for col in feature_columns:
        unique_vals = set(pd.unique(df[col]))
        if not unique_vals.issubset({0, 1}):
            non_binary_columns.append((col, list(unique_vals)[:5]))

    if non_binary_columns:
        raise ValueError(
            f"Expected all {len(feature_columns)} symptom columns to be binary (0/1). "
            f"Found {len(non_binary_columns)} non-binary columns, e.g.: {non_binary_columns[:5]}"
        )

    # --- Shape / class-count sanity warnings (non-fatal) -----------------------
    n_samples, n_total_cols = df.shape
    n_features = len(feature_columns)
    n_classes = df[target_column].nunique()

    if n_samples != expected_n_samples:
        logger.warning(
            "Row count (%d) differs from documented expectation (%d). Continuing anyway.",
            n_samples, expected_n_samples,
        )
    if n_features != expected_n_features:
        logger.warning(
            "Feature column count (%d) differs from documented expectation (%d). Continuing anyway.",
            n_features, expected_n_features,
        )
    if n_classes != expected_n_classes:
        logger.warning(
            "Disease class count (%d) differs from documented expectation (%d). Continuing anyway.",
            n_classes, expected_n_classes,
        )

    logger.info(
        "Dataset validated successfully: %d samples, %d feature columns, %d disease classes.",
        n_samples, n_features, n_classes,
    )
    return feature_columns, n_classes


def load_and_validate(path: str = config.RAW_DATA_PATH) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience wrapper that loads the dataset and runs full validation.

    Args:
        path: Path to the CSV file.

    Returns:
        A tuple of (DataFrame, feature_columns).
    """
    df = load_raw_dataset(path)
    feature_columns, _ = validate_dataset(df)
    return df, feature_columns
