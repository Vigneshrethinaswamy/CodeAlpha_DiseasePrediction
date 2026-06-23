"""
src package
============
Core package for the CodeAlpha_DiseasePrediction project.

Modules:
    config       - Centralized paths, hyperparameters, and constants.
    utils        - Logging setup and shared helper functions.
    data_loader  - Dataset loading and structural validation.
    preprocess   - Target encoding, train/test split, feature scaling.
    eda          - Exploratory data analysis plot generation.
    model        - Model factory functions (Logistic Regression, Random
                   Forest, XGBoost).
    train        - Training orchestration and best-model selection.
    evaluate     - Metric computation and confusion matrix plotting.
    predict      - Inference module for predicting disease from symptoms.
"""
