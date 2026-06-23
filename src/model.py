"""
model.py
========
Model factory functions for the three classifiers compared in this project:
    A. Logistic Regression
    B. Random Forest
    C. XGBoost

Each factory function returns an UNFITTED estimator configured with the
hyperparameters specified in `config.py`. Keeping model construction
separate from training logic (train.py) makes it easy to swap or tune
individual models without touching the training loop.

IMPORTANT scikit-learn compatibility note:
    The `multi_class` parameter of `LogisticRegression` was deprecated in
    scikit-learn 1.5 and REMOVED in scikit-learn >= 1.7 (passing it raises
    a TypeError). This project's installed scikit-learn version is 1.8.0,
    so `multi_class` is intentionally NOT passed. The "lbfgs" solver used
    here natively supports multinomial (softmax) loss for multi-class
    targets without needing the parameter -- the original requirement of
    multinomial behavior is preserved automatically.
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost is not installed. XGBoost model will be unavailable. "
        "Install it with: pip install xgboost"
    )


def build_logistic_regression() -> LogisticRegression:
    """
    Build an unfitted Logistic Regression classifier for multi-class
    disease prediction.

    Configuration (see config.LOGISTIC_REGRESSION_PARAMS):
        - max_iter=2000 (sufficient iterations for convergence on 230 features)
        - solver="lbfgs" (natively supports multinomial/softmax loss for
          multi-class targets; replaces the removed `multi_class` parameter)
        - random_state=42 (reproducibility)
        Note: `n_jobs` is intentionally omitted -- scikit-learn >= 1.8
        treats it as a no-op for LogisticRegression's "lbfgs" solver and
        emits a FutureWarning if passed.

    Returns:
        An unfitted `sklearn.linear_model.LogisticRegression` instance.
    """
    model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    logger.info("Built Logistic Regression model with params: %s", config.LOGISTIC_REGRESSION_PARAMS)
    return model


def build_random_forest() -> RandomForestClassifier:
    """
    Build an unfitted Random Forest classifier for multi-class disease
    prediction.

    Configuration (see config.RANDOM_FOREST_PARAMS):
        - n_estimators=300
        - random_state=42
        - n_jobs=-1

    Returns:
        An unfitted `sklearn.ensemble.RandomForestClassifier` instance.
    """
    model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    logger.info("Built Random Forest model with params: %s", config.RANDOM_FOREST_PARAMS)
    return model


def build_xgboost(num_classes: int) -> Any:
    """
    Build an unfitted XGBoost classifier for multi-class disease prediction.

    Configuration (see config.XGBOOST_PARAMS):
        - n_estimators=300
        - learning_rate=0.1
        - max_depth=8
        - subsample=0.8
        - colsample_bytree=0.8
        - random_state=42
        - objective="multi:softprob" and num_class=<num_classes> are set
          explicitly to ensure proper multi-class probability outputs
          (required for ROC-AUC OvR evaluation and confidence scores in
          predict.py).

    Args:
        num_classes: Number of distinct disease classes in the target.

    Returns:
        An unfitted `xgboost.XGBClassifier` instance.

    Raises:
        ImportError: If the `xgboost` package is not installed.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed in this environment. "
            "Install it with: pip install xgboost"
        )

    params = dict(config.XGBOOST_PARAMS)
    params["objective"] = "multi:softprob"
    params["num_class"] = num_classes

    model = XGBClassifier(**params)
    logger.info("Built XGBoost model with params: %s", params)
    return model


def get_model_registry(num_classes: int) -> dict:
    """
    Construct a registry of all available models keyed by display name.

    If XGBoost is not installed, it is silently omitted from the registry
    (with a warning already logged at import time) so the rest of the
    pipeline (Logistic Regression + Random Forest) can still run end-to-end.

    Args:
        num_classes: Number of distinct disease classes in the target.

    Returns:
        A dictionary mapping model name -> unfitted estimator instance.
        Each entry also records whether the model expects scaled features.
    """
    registry = {
        "Logistic Regression": {
            "model": build_logistic_regression(),
            "needs_scaling": True,
        },
        "Random Forest": {
            "model": build_random_forest(),
            "needs_scaling": False,
        },
    }

    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = {
            "model": build_xgboost(num_classes),
            "needs_scaling": False,
        }
    else:
        logger.warning("Skipping XGBoost in model registry: package not installed.")

    return registry
