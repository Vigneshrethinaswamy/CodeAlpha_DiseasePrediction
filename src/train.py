"""
train.py
========
Training orchestration for the disease prediction project.

Trains all available models (Logistic Regression, Random Forest, and
XGBoost if installed) on the preprocessed dataset, evaluates each one
using `evaluate.py`, and selects the best-performing model based on
weighted F1-score.

The best model, label encoder, and scaler are persisted to `models/` via
joblib, along with a JSON metadata file summarizing all model results for
later inspection (used by README generation and result reporting).
"""

from typing import Any, Dict

from src import config
from src.evaluate import evaluate_model
from src.model import get_model_registry
from src.utils import get_logger, save_joblib, save_json, timer

logger = get_logger(__name__)


def train_single_model(
    model_name: str,
    model: Any,
    needs_scaling: bool,
    X_train: Any,
    X_train_scaled: Any,
    y_train: Any,
) -> Any:
    """
    Fit a single model on the appropriate (scaled or unscaled) training data.

    Args:
        model_name: Display name of the model (for logging).
        model: An unfitted estimator instance.
        needs_scaling: Whether this model should be trained on scaled features.
        X_train: Unscaled training feature matrix (DataFrame or array).
        X_train_scaled: Scaled training feature matrix (numpy array).
        y_train: Encoded training labels.

    Returns:
        The fitted model.
    """
    training_data = X_train_scaled if needs_scaling else X_train
    with timer(f"Training {model_name}"):
        model.fit(training_data, y_train)
    return model


def train_and_evaluate_all(preprocessed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Train all registered models and evaluate each one on the held-out test set.

    Args:
        preprocessed: The dictionary returned by `preprocess.run_preprocessing`,
            containing X_train, X_test, X_train_scaled, X_test_scaled,
            y_train, y_test, label_encoder, scaler, and feature_columns.

    Returns:
        A dictionary mapping model name -> {
            'model': fitted estimator,
            'needs_scaling': bool,
            'metrics': metrics dict from evaluate.evaluate_model
        }
    """
    label_encoder = preprocessed["label_encoder"]
    class_names = [str(c) for c in label_encoder.classes_]
    num_classes = len(class_names)

    registry = get_model_registry(num_classes)
    results: Dict[str, Dict[str, Any]] = {}

    for model_name, entry in registry.items():
        model = entry["model"]
        needs_scaling = entry["needs_scaling"]

        try:
            fitted_model = train_single_model(
                model_name=model_name,
                model=model,
                needs_scaling=needs_scaling,
                X_train=preprocessed["X_train"],
                X_train_scaled=preprocessed["X_train_scaled"],
                y_train=preprocessed["y_train"],
            )

            X_test_for_eval = (
                preprocessed["X_test_scaled"] if needs_scaling else preprocessed["X_test"]
            )
            metrics = evaluate_model(
                model=fitted_model,
                model_name=model_name,
                X_test=X_test_for_eval,
                y_test=preprocessed["y_test"],
                class_names=class_names,
            )

            results[model_name] = {
                "model": fitted_model,
                "needs_scaling": needs_scaling,
                "metrics": metrics,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Training/evaluation failed for model '%s': %s", model_name, exc)
            continue

    if not results:
        raise RuntimeError(
            "All models failed to train. Check the logs above for the root cause."
        )

    return results


def select_best_model(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Select the best-performing model based on weighted F1-score.

    Args:
        results: The dictionary returned by `train_and_evaluate_all`.

    Returns:
        The name (key) of the best model in `results`.
    """
    best_model_name = max(
        results.keys(), key=lambda name: results[name]["metrics"]["f1_weighted"]
    )
    best_f1 = results[best_model_name]["metrics"]["f1_weighted"]
    logger.info(
        "Best model selected: '%s' with weighted F1-score = %.4f", best_model_name, best_f1
    )
    return best_model_name


def save_best_model_artifacts(
    results: Dict[str, Dict[str, Any]],
    best_model_name: str,
    label_encoder: Any,
    scaler: Any,
) -> None:
    """
    Persist the best model and supporting artifacts to `models/`.

    Saves:
        - models/best_model.pkl    (best fitted estimator)
        - models/label_encoder.pkl (already saved by preprocess.py, re-saved
                                     here for completeness/idempotency)
        - models/scaler.pkl        (already saved by preprocess.py, re-saved
                                     here for completeness/idempotency)
        - models/model_metadata.json (summary of all models' metrics + which
                                       one was selected, plus whether the
                                       selected model needs scaled inputs)

    Args:
        results: The dictionary returned by `train_and_evaluate_all`.
        best_model_name: Name of the selected best model.
        label_encoder: The fitted LabelEncoder used for the target column.
        scaler: The fitted StandardScaler used for feature scaling.
    """
    best_entry = results[best_model_name]
    save_joblib(best_entry["model"], config.BEST_MODEL_PATH)
    save_joblib(label_encoder, config.LABEL_ENCODER_PATH)
    save_joblib(scaler, config.SCALER_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "best_model_needs_scaling": best_entry["needs_scaling"],
        "models": {
            name: {
                "accuracy": entry["metrics"]["accuracy"],
                "precision_weighted": entry["metrics"]["precision_weighted"],
                "recall_weighted": entry["metrics"]["recall_weighted"],
                "f1_weighted": entry["metrics"]["f1_weighted"],
                "roc_auc_ovr": entry["metrics"]["roc_auc_ovr"],
                "needs_scaling": entry["needs_scaling"],
            }
            for name, entry in results.items()
        },
    }
    save_json(metadata, config.MODEL_METADATA_PATH)
    logger.info(
        "Saved best model ('%s') and supporting artifacts to '%s'.",
        best_model_name, config.MODELS_DIR,
    )
