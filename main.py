"""
main.py
=======
Top-level orchestration script for the CodeAlpha_DiseasePrediction project.

Executes the full end-to-end pipeline in order:

    1. Ensure required directories exist.
    2. Load and validate the raw dataset.
    3. Run exploratory data analysis (saves plots to plots/).
    4. Preprocess the data (encode target, split, scale; saves encoder
       and scaler to models/).
    5. Train all available models (Logistic Regression, Random Forest,
       XGBoost) and evaluate each on the held-out test set (saves
       confusion matrices to plots/).
    6. Select the best model by weighted F1-score.
    7. Persist the best model and supporting artifacts to models/.
    8. Print a final summary of all model results to the console.

Run from the project root with:
    python main.py
"""

import sys
import time

from src import config
from src.data_loader import load_and_validate
from src.eda import run_full_eda
from src.preprocess import run_preprocessing
from src.train import save_best_model_artifacts, select_best_model, train_and_evaluate_all
from src.utils import format_seconds, get_logger, save_json

logger = get_logger(__name__)


def print_summary(results: dict, best_model_name: str) -> None:
    """
    Print a human-readable summary table of all trained models' metrics
    to the console, with the selected best model highlighted.

    Args:
        results: The dictionary returned by `train_and_evaluate_all`.
        best_model_name: Name of the model selected as best.
    """
    print("\n" + "=" * 78)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 78)
    header = f"{'Model':<22}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1 (W)':>10}{'ROC-AUC':>10}"
    print(header)
    print("-" * 78)
    for name, entry in results.items():
        m = entry["metrics"]
        marker = " *" if name == best_model_name else ""
        print(
            f"{name:<22}{m['accuracy']:>10.4f}{m['precision_weighted']:>12.4f}"
            f"{m['recall_weighted']:>10.4f}{m['f1_weighted']:>10.4f}{m['roc_auc_ovr']:>10.4f}{marker}"
        )
    print("-" * 78)
    print(f"* Best model (highest weighted F1-score): {best_model_name}")
    print("=" * 78 + "\n")


def run_pipeline() -> None:
    """
    Execute the complete disease prediction pipeline end-to-end.

    Raises:
        Exception: Re-raises any unrecoverable error encountered during
            the pipeline after logging it, so the process exits with a
            non-zero status code for visibility in CI / shell scripting.
    """
    pipeline_start = time.time()
    logger.info("=" * 78)
    logger.info("Starting CodeAlpha_DiseasePrediction pipeline")
    logger.info("=" * 78)

    config.ensure_directories()

    # -------------------------------------------------------------------
    # Step 1: Load and validate data
    # -------------------------------------------------------------------
    logger.info("STEP 1/6: Loading and validating dataset...")
    df, feature_columns = load_and_validate()
    save_json(feature_columns, config.FEATURE_COLUMNS_PATH)

    # -------------------------------------------------------------------
    # Step 2: Exploratory Data Analysis
    # -------------------------------------------------------------------
    logger.info("STEP 2/6: Running exploratory data analysis...")
    run_full_eda(df, feature_columns)

    # -------------------------------------------------------------------
    # Step 3: Preprocessing
    # -------------------------------------------------------------------
    logger.info("STEP 3/6: Preprocessing data (encoding, splitting, scaling)...")
    preprocessed = run_preprocessing(df, feature_columns)

    # -------------------------------------------------------------------
    # Step 4: Training & Evaluation
    # -------------------------------------------------------------------
    logger.info("STEP 4/6: Training and evaluating all models...")
    results = train_and_evaluate_all(preprocessed)

    # -------------------------------------------------------------------
    # Step 5: Best Model Selection
    # -------------------------------------------------------------------
    logger.info("STEP 5/6: Selecting best model...")
    best_model_name = select_best_model(results)

    # -------------------------------------------------------------------
    # Step 6: Save Artifacts
    # -------------------------------------------------------------------
    logger.info("STEP 6/6: Saving best model and supporting artifacts...")
    save_best_model_artifacts(
        results=results,
        best_model_name=best_model_name,
        label_encoder=preprocessed["label_encoder"],
        scaler=preprocessed["scaler"],
    )

    print_summary(results, best_model_name)

    elapsed = time.time() - pipeline_start
    logger.info("Pipeline completed successfully in %s.", format_seconds(elapsed))
    logger.info("Artifacts saved to: %s", config.MODELS_DIR)
    logger.info("Plots saved to: %s", config.PLOTS_DIR)
    logger.info(
        "Run predictions with: python -m src.predict --symptoms \"<symptom1,symptom2,...>\""
    )


def main() -> None:
    """
    Entry point with top-level error handling: logs unhandled exceptions
    with a full traceback and exits with status code 1 so failures are
    visible to shell scripts / CI pipelines, rather than failing silently.
    """
    try:
        run_pipeline()
    except Exception:
        logger.exception("Pipeline failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
