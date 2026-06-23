"""
evaluate.py
===========
Evaluation utilities for trained disease-prediction models.

For every model, computes:
    - Accuracy
    - Precision (weighted)
    - Recall (weighted)
    - F1-score (weighted)
    - ROC-AUC (One-vs-Rest)
    - Full classification report (per-class precision/recall/F1)
    - Confusion matrix (saved as a plot to `plots/`)

Design notes:
    - ROC-AUC (OvR) requires predicted class probabilities, which all
      three models provide via `predict_proba`.
    - For datasets with 100 classes, only classes present in `y_test`
      are passed to `roc_auc_score` with explicit `labels=...` to avoid
      errors from classes with zero test-set representation.
    - Confusion matrices for 100 classes are large; the plot is sized
      dynamically and uses a log-scale-friendly color map for readability.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.utils import get_logger, safe_filename

logger = get_logger(__name__)


def _align_proba_to_full_label_space(
    y_proba: np.ndarray, model_classes: np.ndarray, n_total_classes: int
) -> np.ndarray:
    """
    Re-index a model's predicted probability matrix so that column `i`
    always corresponds to encoded label `i` in the FULL label space.

    This is necessary because `model.classes_` only contains classes that
    were actually present in the training data. If a stratified split is
    not possible (e.g. a disease with only 1 sample) and that class is
    absent from the training fold, `predict_proba` returns a matrix with
    fewer columns than `n_total_classes`, and naively assuming
    `column index == encoded label` silently misaligns every class above
    the missing one. This function builds a zero-filled matrix of shape
    (n_samples, n_total_classes) and scatters each model column into its
    correct global label position.

    Args:
        y_proba: Probability matrix as returned by `model.predict_proba`,
            shape (n_samples, len(model_classes)).
        model_classes: The `classes_` attribute of the fitted model,
            giving the encoded label each column of `y_proba` corresponds to.
        n_total_classes: Total number of classes in the full label space
            (i.e. `len(label_encoder.classes_)`).

    Returns:
        A probability matrix of shape (n_samples, n_total_classes), with
        zero probability for any class absent from the model's training data.
    """
    n_samples = y_proba.shape[0]
    aligned = np.zeros((n_samples, n_total_classes), dtype=float)
    for col_idx, encoded_label in enumerate(model_classes):
        aligned[:, int(encoded_label)] = y_proba[:, col_idx]
    return aligned


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: List[int],
    model_classes: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute the full evaluation metric suite for a single model's predictions.

    Args:
        y_true: Ground-truth encoded labels for the test set.
        y_pred: Predicted encoded labels for the test set.
        y_proba: Predicted class probabilities, shape (n_samples, n_model_classes).
        class_labels: Sorted list of all possible encoded class labels
            (i.e. `range(num_classes)`), used to align `y_proba` columns
            with `roc_auc_score`'s expected label ordering.
        model_classes: The fitted model's `classes_` attribute. If
            provided and its length differs from `len(class_labels)`,
            `y_proba` is re-indexed to the full label space before
            computing ROC-AUC, since some rare classes may be absent from
            the training fold (e.g. via a non-stratified split fallback).

    Returns:
        A dictionary containing all computed metrics:
            'accuracy', 'precision_weighted', 'recall_weighted',
            'f1_weighted', 'roc_auc_ovr', 'classification_report' (dict),
            'confusion_matrix' (numpy array).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Align probability columns to the full label space if the model was
    # trained on a subset of classes (e.g. a rare class missing from the
    # training fold after a non-stratified split fallback).
    if model_classes is not None and len(model_classes) != len(class_labels):
        logger.warning(
            "Model was trained on %d of %d total classes (some classes had too "
            "few samples to appear in the training fold). Aligning probability "
            "columns to the full label space for ROC-AUC computation.",
            len(model_classes), len(class_labels),
        )
        y_proba_aligned = _align_proba_to_full_label_space(
            y_proba, model_classes, len(class_labels)
        )
    else:
        y_proba_aligned = y_proba

    # ROC-AUC (OvR) requires probability columns aligned with `labels`.
    # Restrict to classes actually present in y_true to avoid sklearn errors
    # when a class has zero samples in the test split.
    present_labels = sorted(np.unique(y_true).tolist())
    try:
        roc_auc = roc_auc_score(
            y_true,
            y_proba_aligned,
            average="weighted",
            multi_class="ovr",
            labels=class_labels,
        )
    except ValueError as exc:
        logger.warning(
            "Full ROC-AUC computation failed (%s). Retrying with only classes "
            "present in the test set (%d of %d classes).",
            exc, len(present_labels), len(class_labels),
        )
        try:
            proba_subset = y_proba_aligned[:, present_labels]
            # Re-normalize subset probabilities so each row sums to 1.
            row_sums = proba_subset.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            proba_subset = proba_subset / row_sums
            roc_auc = roc_auc_score(
                y_true, proba_subset, average="weighted", multi_class="ovr", labels=present_labels
            )
        except ValueError as exc2:
            logger.error("ROC-AUC computation failed entirely: %s. Reporting as NaN.", exc2)
            roc_auc = float("nan")

    report_dict = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "roc_auc_ovr": float(roc_auc),
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: List[str],
    max_labeled_classes: int = 40,
) -> str:
    """
    Plot and save a confusion matrix heatmap for a given model.

    With up to 100 classes, individual cell annotations and full tick
    labels become unreadable. If the number of classes exceeds
    `max_labeled_classes`, tick labels are omitted (axes are still
    correctly ordered) to keep the plot legible; otherwise full disease
    names are shown.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes).
        model_name: Display name of the model (used in title and filename).
        class_names: List of original disease class names, ordered to
            match the encoded label indices (i.e. `label_encoder.classes_`).
        max_labeled_classes: Threshold above which tick labels are hidden.

    Returns:
        Path to the saved confusion matrix plot.
    """
    n_classes = cm.shape[0]
    show_labels = n_classes <= max_labeled_classes

    fig_size = max(8, n_classes * 0.22)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Use log scale coloring since diagonal (correct) counts dominate and
    # would otherwise wash out off-diagonal (error) cells visually.
    cm_display = np.log1p(cm)

    sns.heatmap(
        cm_display,
        ax=ax,
        cmap="Blues",
        square=True,
        cbar_kws={"shrink": 0.7, "label": "log(1 + count)"},
        xticklabels=class_names if show_labels else False,
        yticklabels=class_names if show_labels else False,
    )
    ax.set_xlabel("Predicted Disease")
    ax.set_ylabel("True Disease")
    ax.set_title(f"Confusion Matrix — {model_name}")

    if show_labels:
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    filename = f"confusion_matrix_{safe_filename(model_name)}.png"
    output_path = os.path.join(config.PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot: %s", output_path)
    return output_path


def evaluate_model(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Run full evaluation for a single fitted model: compute metrics, log a
    summary, and save the confusion matrix plot.

    Args:
        model: A fitted classifier exposing `predict` and `predict_proba`.
        model_name: Display name of the model.
        X_test: Test feature matrix (already scaled if the model requires it).
        y_test: Encoded true labels for the test set.
        class_names: Original disease class names ordered by encoded label index.

    Returns:
        The metrics dictionary produced by `compute_metrics`, with an
        additional 'confusion_matrix_path' key.
    """
    logger.info("Evaluating model: %s", model_name)
    class_labels = list(range(len(class_names)))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    model_classes = getattr(model, "classes_", None)

    metrics = compute_metrics(y_test, y_pred, y_proba, class_labels, model_classes=model_classes)
    cm_path = plot_confusion_matrix(metrics["confusion_matrix"], model_name, class_names)
    metrics["confusion_matrix_path"] = cm_path

    logger.info(
        "%s — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC (OvR): %.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"],
        metrics["roc_auc_ovr"],
    )
    return metrics
