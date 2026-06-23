"""
preprocess.py
=============
Preprocessing pipeline for the disease prediction dataset:
    1. Label-encode the multi-class target column.
    2. Stratified train/test split.
    3. Feature scaling (StandardScaler) for models that require it
       (Logistic Regression). Tree-based models (Random Forest, XGBoost)
       use the unscaled features directly.

All fitted transformers (LabelEncoder, StandardScaler) are persisted with
joblib so that predict.py can apply identical transformations at inference
time.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config
from src.utils import get_logger, save_joblib

logger = get_logger(__name__)


def encode_target(
    df: pd.DataFrame, target_column: str = config.TARGET_COLUMN
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Label-encode the categorical target column into integer class indices.

    Args:
        df: The dataset containing the target column.
        target_column: Name of the target column to encode.

    Returns:
        A tuple of (encoded_labels, fitted LabelEncoder).
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df[target_column].values)
    logger.info("Encoded target column '%s' into %d classes.", target_column, len(encoder.classes_))
    return y_encoded, encoder


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Perform a stratified train/test split.

    If stratification fails because one or more classes have too few
    samples to be represented in both splits (scikit-learn requires at
    least 2 members per class for a stratified split), the function logs
    a warning and falls back to a non-stratified split rather than
    crashing the pipeline.

    Args:
        X: Feature matrix.
        y: Encoded target array.
        test_size: Fraction of data to reserve for testing.
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        logger.warning(
            "At least one class has only %d sample(s), which is too few for a "
            "stratified split. Falling back to a non-stratified split.",
            min_class_count,
        )
        stratify_arg = None
    else:
        stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    logger.info(
        "Split data into train (%d rows) and test (%d rows) sets.",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the training features and apply it to both
    train and test sets.

    The scaler is fit ONLY on training data to prevent data leakage from
    the test set into the preprocessing step.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        A tuple of (X_train_scaled, X_test_scaled, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Fitted StandardScaler on training data (%d features).", X_train.shape[1])
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = config.TARGET_COLUMN,
) -> dict:
    """
    Execute the full preprocessing pipeline and persist fitted artifacts.

    Steps:
        1. Encode the target column.
        2. Stratified train/test split on raw (unscaled) features.
        3. Fit a StandardScaler on the training split.
        4. Save the LabelEncoder and StandardScaler to disk via joblib.

    Args:
        df: The validated raw dataset.
        feature_columns: List of symptom feature column names.
        target_column: Name of the target column.

    Returns:
        A dictionary with keys:
            'X_train', 'X_test'               -> unscaled DataFrames (for tree models)
            'X_train_scaled', 'X_test_scaled'  -> numpy arrays (for Logistic Regression)
            'y_train', 'y_test'                -> encoded integer label arrays
            'label_encoder', 'scaler'          -> fitted transformer objects
            'feature_columns'                  -> list of feature column names used
    """
    X = df[feature_columns].astype(np.float32)
    y, label_encoder = encode_target(df, target_column)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    save_joblib(label_encoder, config.LABEL_ENCODER_PATH)
    save_joblib(scaler, config.SCALER_PATH)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
