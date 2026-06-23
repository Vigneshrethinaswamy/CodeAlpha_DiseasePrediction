"""
eda.py
======
Exploratory Data Analysis for the disease-symptom dataset.

Generates and saves the following plots into `plots/`:
    1. Class distribution (all 100 diseases, sorted by frequency)
    2. Top-N most frequent diseases
    3. Symptom (feature) frequency (top-N most common symptoms)
    4. Correlation heatmap (subsampled features, for readability)
    5. Feature sparsity visualization (how many symptoms are active per sample)

All plotting functions are side-effecting (they save a PNG to disk) and
return the output path, so `main.py` can log exactly what was produced.
Each function is independent and defensive: a failure generating one plot
(e.g. due to an unusual data distribution) does not halt the others.
"""

import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; required for headless execution.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def _savefig(fig: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib figure to the plots directory and close it to free memory.

    Args:
        fig: The matplotlib Figure to save.
        filename: Filename (including extension) to save under `config.PLOTS_DIR`.

    Returns:
        The full path the figure was saved to.
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(config.PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)
    return output_path


def plot_class_distribution(df: pd.DataFrame, target_column: str = config.TARGET_COLUMN) -> str:
    """
    Plot the full distribution of disease classes, sorted by frequency.

    Useful for visually inspecting class imbalance across all 100 diseases.

    Args:
        df: The dataset.
        target_column: Name of the target column.

    Returns:
        Path to the saved plot.
    """
    counts = df[target_column].value_counts()
    fig, ax = plt.subplots(figsize=(16, max(8, len(counts) * 0.18)))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, ax=ax,
                palette="viridis", legend=False)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Disease")
    ax.set_title(f"Class Distribution Across All {len(counts)} Disease Classes")
    return _savefig(fig, "class_distribution.png")


def plot_top_diseases(
    df: pd.DataFrame, target_column: str = config.TARGET_COLUMN, top_n: int = config.TOP_N_DISEASES
) -> str:
    """
    Plot the top-N most frequent diseases in the dataset.

    Args:
        df: The dataset.
        target_column: Name of the target column.
        top_n: Number of top diseases to display.

    Returns:
        Path to the saved plot.
    """
    counts = df[target_column].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, ax=ax,
                palette="mako", legend=False)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Disease")
    ax.set_title(f"Top {top_n} Most Frequent Diseases")
    return _savefig(fig, "top_diseases.png")


def plot_symptom_frequency(
    df: pd.DataFrame, feature_columns: List[str], top_n: int = config.TOP_N_SYMPTOMS
) -> str:
    """
    Plot the top-N most frequently occurring symptoms across all samples.

    Since symptom columns are binary, frequency is computed as the column
    sum (i.e. how many samples report that symptom as present).

    Args:
        df: The dataset.
        feature_columns: List of symptom column names.
        top_n: Number of top symptoms to display.

    Returns:
        Path to the saved plot.
    """
    symptom_counts = df[feature_columns].sum(axis=0).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x=symptom_counts.values, y=symptom_counts.index, hue=symptom_counts.index,
                ax=ax, palette="flare", legend=False)
    ax.set_xlabel("Number of Samples Reporting Symptom")
    ax.set_ylabel("Symptom")
    ax.set_title(f"Top {top_n} Most Frequent Symptoms")
    return _savefig(fig, "symptom_frequency.png")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_columns: List[str],
    max_features: int = config.CORRELATION_HEATMAP_MAX_FEATURES,
    random_state: int = config.RANDOM_STATE,
) -> str:
    """
    Plot a correlation heatmap of a subsample of symptom features.

    With 230 binary features, a full correlation matrix is too dense to
    visually interpret. This function selects the `max_features` most
    frequent symptoms (which are also the most statistically informative
    to visualize) and plots their pairwise correlation.

    Args:
        df: The dataset.
        feature_columns: List of symptom column names.
        max_features: Maximum number of features to include in the heatmap.
        random_state: Seed (unused for selection logic but kept for API
            consistency with the rest of the pipeline).

    Returns:
        Path to the saved plot.
    """
    symptom_counts = df[feature_columns].sum(axis=0).sort_values(ascending=False)
    top_features = symptom_counts.head(max_features).index.tolist()

    corr_matrix = df[top_features].corr()
    fig, ax = plt.subplots(figsize=(max(10, max_features * 0.35), max(8, max_features * 0.35)))
    sns.heatmap(
        corr_matrix, ax=ax, cmap="coolwarm", center=0,
        square=True, linewidths=0.3, cbar_kws={"shrink": 0.7},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(f"Correlation Heatmap (Top {max_features} Most Frequent Symptoms)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    return _savefig(fig, "correlation_heatmap.png")


def plot_feature_sparsity(df: pd.DataFrame, feature_columns: List[str]) -> str:
    """
    Visualize feature sparsity: the distribution of how many symptoms are
    "active" (value == 1) per sample.

    Since each row represents a patient case with a sparse subset of 230
    possible symptoms marked, this plot reveals how many symptoms are
    typically reported per case.

    Args:
        df: The dataset.
        feature_columns: List of symptom column names.

    Returns:
        Path to the saved plot.
    """
    active_symptom_counts = df[feature_columns].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(active_symptom_counts, bins=40, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Number of Active Symptoms per Sample")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Active Symptoms per Sample")

    overall_density = active_symptom_counts.sum() / (df.shape[0] * len(feature_columns)) * 100
    sns.boxplot(x=active_symptom_counts, ax=axes[1], color="lightcoral")
    axes[1].set_xlabel("Number of Active Symptoms per Sample")
    axes[1].set_title(f"Sparsity Overview (Overall Density: {overall_density:.2f}%)")

    return _savefig(fig, "feature_sparsity.png")


def run_full_eda(df: pd.DataFrame, feature_columns: List[str]) -> dict:
    """
    Run the complete EDA suite and save all plots to `plots/`.

    Each plot is generated independently with its own try/except so that
    a failure in one visualization (e.g. due to an edge case in the data)
    does not prevent the rest of the EDA suite from completing.

    Args:
        df: The validated dataset.
        feature_columns: List of symptom feature column names.

    Returns:
        A dictionary mapping plot names to their saved file paths (or
        None if that specific plot failed to generate).
    """
    logger.info("Starting full EDA suite...")
    results = {}

    plot_jobs = [
        ("class_distribution", lambda: plot_class_distribution(df)),
        ("top_diseases", lambda: plot_top_diseases(df)),
        ("symptom_frequency", lambda: plot_symptom_frequency(df, feature_columns)),
        ("correlation_heatmap", lambda: plot_correlation_heatmap(df, feature_columns)),
        ("feature_sparsity", lambda: plot_feature_sparsity(df, feature_columns)),
    ]

    for plot_name, plot_fn in plot_jobs:
        try:
            results[plot_name] = plot_fn()
        except Exception as exc:  # noqa: BLE001 - EDA plots must not crash the pipeline
            logger.error("Failed to generate plot '%s': %s", plot_name, exc)
            results[plot_name] = None

    logger.info("EDA suite complete. Generated %d/%d plots successfully.",
                sum(1 for v in results.values() if v is not None), len(plot_jobs))
    return results
