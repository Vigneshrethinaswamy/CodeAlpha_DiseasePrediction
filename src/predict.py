"""
predict.py
==========
Inference module for the disease prediction pipeline.

Loads the persisted best model, label encoder, and scaler, then exposes a
simple interface for predicting a disease from a list of symptom names.

Usage example (programmatic):
    from src.predict import DiseasePredictor

    predictor = DiseasePredictor()
    result = predictor.predict_from_symptoms(["fever", "cough", "headache"])
    print(result["predicted_disease"], result["confidence"])

Usage example (command line):
    python -m src.predict --symptoms "fever,cough,headache"
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, load_joblib

logger = get_logger(__name__)


class DiseasePredictor:
    """
    Loads trained artifacts (best model, label encoder, scaler, and feature
    column list) and provides an interface for predicting diseases from
    symptom inputs.

    Attributes:
        model: The fitted best-performing classifier.
        label_encoder: Fitted LabelEncoder for decoding predicted class
            indices back into disease names.
        scaler: Fitted StandardScaler, applied only if the best model was
            trained on scaled features (e.g. Logistic Regression).
        feature_columns: Ordered list of all 230 symptom column names,
            used to build the binary feature vector from a symptom list.
        needs_scaling: Whether the loaded best model requires scaled input.
    """

    def __init__(
        self,
        model_path: str = config.BEST_MODEL_PATH,
        encoder_path: str = config.LABEL_ENCODER_PATH,
        scaler_path: str = config.SCALER_PATH,
        metadata_path: str = config.MODEL_METADATA_PATH,
        feature_columns_path: str = config.FEATURE_COLUMNS_PATH,
    ) -> None:
        """
        Initialize the predictor by loading all required artifacts from disk.

        Args:
            model_path: Path to the saved best model (.pkl).
            encoder_path: Path to the saved LabelEncoder (.pkl).
            scaler_path: Path to the saved StandardScaler (.pkl).
            metadata_path: Path to the model metadata JSON (used to
                determine whether the best model requires scaled input).
            feature_columns_path: Path to the JSON file listing all
                symptom feature column names in their training order.

        Raises:
            FileNotFoundError: If any required artifact is missing, with a
                message indicating that the training pipeline must be run first.
        """
        self.model = load_joblib(model_path)
        self.label_encoder = load_joblib(encoder_path)
        self.scaler = load_joblib(scaler_path)
        self.feature_columns = self._load_feature_columns(feature_columns_path)
        self.needs_scaling = self._load_needs_scaling(metadata_path)

        self._symptom_lookup = {col.lower().strip(): col for col in self.feature_columns}

        logger.info(
            "DiseasePredictor initialized with %d known symptoms and %d disease classes.",
            len(self.feature_columns), len(self.label_encoder.classes_),
        )

    @staticmethod
    def _load_feature_columns(path: str) -> List[str]:
        """
        Load the ordered list of feature (symptom) column names.

        Args:
            path: Path to the JSON file containing the feature column list.

        Returns:
            A list of symptom column names in training order.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature columns file not found at '{path}'. "
                f"Please run the training pipeline (main.py) first."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_needs_scaling(path: str) -> bool:
        """
        Determine whether the best model requires scaled input features by
        reading the model metadata JSON produced during training.

        Args:
            path: Path to the model metadata JSON file.

        Returns:
            True if the best model needs scaled features, False otherwise.
            Defaults to False if metadata is unavailable.
        """
        if not os.path.exists(path):
            logger.warning(
                "Model metadata not found at '%s'. Defaulting needs_scaling=False.", path
            )
            return False
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return bool(metadata.get("best_model_needs_scaling", False))

    def _build_feature_vector(self, symptoms: List[str]) -> pd.DataFrame:
        """
        Convert a list of symptom names into a single-row binary feature
        DataFrame matching the model's expected input schema.

        Symptom names are matched case-insensitively and with surrounding
        whitespace stripped. Unrecognized symptom names are logged as
        warnings and silently ignored (they contribute no signal, but
        should not crash inference).

        Args:
            symptoms: List of symptom names provided by the caller.

        Returns:
            A single-row pandas DataFrame with one binary column per
            known symptom, ordered to match `self.feature_columns`.
        """
        vector = {col: 0 for col in self.feature_columns}
        unrecognized = []

        for raw_symptom in symptoms:
            key = raw_symptom.lower().strip()
            matched_col = self._symptom_lookup.get(key)
            if matched_col is not None:
                vector[matched_col] = 1
            else:
                unrecognized.append(raw_symptom)

        if unrecognized:
            logger.warning(
                "%d symptom(s) not recognized and were ignored: %s",
                len(unrecognized), unrecognized,
            )

        return pd.DataFrame([vector], columns=self.feature_columns)

    def predict_from_symptoms(
        self, symptoms: List[str], top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict the most likely disease(s) given a list of reported symptoms.

        Args:
            symptoms: List of symptom names (case-insensitive). Names not
                found in the known symptom vocabulary are ignored with a
                logged warning.
            top_k: Number of top candidate diseases (with confidence
                scores) to include in the result, in addition to the
                single best prediction.

        Returns:
            A dictionary with keys:
                'predicted_disease': str, the single most likely disease.
                'confidence': float, probability assigned to the top prediction.
                'top_k_predictions': List[Dict[str, Any]], each containing
                    'disease' and 'confidence', sorted descending.
                'recognized_symptoms': List[str], symptoms that were matched.
                'unrecognized_symptoms': List[str], symptoms that were ignored.

        Raises:
            ValueError: If `symptoms` is empty or contains no recognizable
                symptom names (a prediction would be meaningless / all-zero).
        """
        if not symptoms:
            raise ValueError("At least one symptom must be provided for prediction.")

        feature_vector = self._build_feature_vector(symptoms)

        recognized = [c for c in self.feature_columns if feature_vector.iloc[0][c] == 1]
        unrecognized = [s for s in symptoms if s.lower().strip() not in self._symptom_lookup]

        if not recognized:
            raise ValueError(
                "None of the provided symptoms were recognized. "
                f"Examples of valid symptoms: {self.feature_columns[:5]}"
            )

        model_input = self.scaler.transform(feature_vector) if self.needs_scaling else feature_vector

        probabilities = self.model.predict_proba(model_input)[0]
        top_k = min(top_k, len(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        top_k_predictions = [
            {
                "disease": self.label_encoder.inverse_transform([idx])[0],
                "confidence": float(probabilities[idx]),
            }
            for idx in top_indices
        ]

        result = {
            "predicted_disease": top_k_predictions[0]["disease"],
            "confidence": top_k_predictions[0]["confidence"],
            "top_k_predictions": top_k_predictions,
            "recognized_symptoms": recognized,
            "unrecognized_symptoms": unrecognized,
        }

        logger.info(
            "Predicted disease: '%s' (confidence: %.4f) from %d recognized symptom(s).",
            result["predicted_disease"], result["confidence"], len(recognized),
        )
        return result


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone execution of predict.py.

    Returns:
        Parsed argparse.Namespace with a `symptoms` attribute (comma-
        separated string) and a `top_k` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Predict the most likely disease from a list of symptoms."
    )
    parser.add_argument(
        "--symptoms",
        type=str,
        required=True,
        help="Comma-separated list of symptom names, e.g. 'fever,cough,headache'",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top candidate diseases to display (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point: parse arguments, run prediction, and print
    a formatted result summary to stdout (in addition to standard logging).
    """
    args = _parse_args()
    symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]

    predictor = DiseasePredictor()
    result = predictor.predict_from_symptoms(symptoms, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("DISEASE PREDICTION RESULT")
    print("=" * 60)
    print(f"Predicted Disease : {result['predicted_disease']}")
    print(f"Confidence        : {result['confidence']:.2%}")
    print("\nTop-k Candidates:")
    for i, pred in enumerate(result["top_k_predictions"], start=1):
        print(f"  {i}. {pred['disease']:<40s} {pred['confidence']:.2%}")
    if result["unrecognized_symptoms"]:
        print(f"\nUnrecognized symptoms (ignored): {result['unrecognized_symptoms']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
