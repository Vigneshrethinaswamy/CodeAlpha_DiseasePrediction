# Disease Prediction System

An end-to-end machine learning pipeline for **multi-class disease prediction from binary symptom data**.

Given a set of reported symptoms, the system predicts the most likely disease (out of 100 possible classes) along with a confidence score, by comparing three classical ML algorithms and automatically selecting the best performer.

---

## Project Overview

| Property | Value |
|---|---|
| Samples | 96,088 |
| Features | 230 binary symptom indicators |
| Target | `diseases` (100 classes) |
| Missing values | None |
| Problem type | Multi-class classification |
| Models compared | Logistic Regression, Random Forest, XGBoost |
| Selection metric | Weighted F1-score |

---

## Folder Structure

```
Disease-Prediction-System/
│
├── data/
│   ├── raw/                 # Place Diseases_and_Symptoms_dataset.csv here
│   ├── processed/           # Cached feature column list (auto-generated)
│   └── external/            # Reserved for any supplementary data
│
├── models/                  # Saved model artifacts (auto-generated)
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── plots/                   # EDA and evaluation visualizations (auto-generated)
├── notebooks/                # Reserved for exploratory notebooks
│
├── src/
│   ├── config.py            # Centralized paths, hyperparameters, constants
│   ├── data_loader.py       # Dataset loading and structural validation
│   ├── preprocess.py        # Label encoding, train/test split, scaling
│   ├── eda.py                # Exploratory data analysis & plot generation
│   ├── model.py              # Model factory functions
│   ├── train.py               # Training orchestration & best-model selection
│   ├── evaluate.py            # Metric computation & confusion matrices
│   ├── predict.py              # Inference module (symptom → disease)
│   └── utils.py                # Logging and shared helpers
│
├── main.py                  # Full pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Architecture

```
Load & Validate Data
        ↓
Exploratory Data Analysis  (plots/)
        ↓
Preprocessing
   - Label encode `diseases`
   - Stratified train/test split (random_state=42)
   - StandardScaler fit on train only
        ↓
Train Models
   - Logistic Regression (max_iter=2000, lbfgs solver)
   - Random Forest (n_estimators=300)
   - XGBoost (n_estimators=300, max_depth=8)
        ↓
Evaluate Models
   - Accuracy, Precision, Recall, F1 (weighted), ROC-AUC (OvR)
   - Confusion matrices (plots/)
        ↓
Select Best Model (by weighted F1-score)
        ↓
Persist Artifacts (models/)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Place the dataset file at:

```
data/raw/Diseases_and_Symptoms_dataset.csv
```

The CSV must contain 230 binary symptom columns plus a `diseases` target column. Column names beyond the target are auto-detected, so the pipeline does not require an exact column count match — it validates structure (no missing values, binary feature values) rather than hardcoding column names.

### 3. Run the full pipeline

```bash
python main.py
```

This single command runs the entire pipeline: EDA → preprocessing → training → evaluation → best-model selection → artifact saving. Console and file logs are written throughout (see `logs/pipeline.log`).

---

## Making Predictions

Once `main.py` has completed at least one full run (so that `models/best_model.pkl` and related artifacts exist), predictions can be made via the command line:

```bash
python -m src.predict --symptoms "fever,cough,headache" --top_k 3
```

**Output:**

```
============================================================
DISEASE PREDICTION RESULT
============================================================
Predicted Disease : <disease name>
Confidence        : XX.XX%

Top-k Candidates:
  1. <disease name>                          XX.XX%
  2. <disease name>                          XX.XX%
  3. <disease name>                          XX.XX%
============================================================
```

### Programmatic usage

```python
from src.predict import DiseasePredictor

predictor = DiseasePredictor()
result = predictor.predict_from_symptoms(["fever", "cough", "headache"], top_k=3)

print(result["predicted_disease"], result["confidence"])
print(result["top_k_predictions"])
```

Symptom names are matched case-insensitively against the 230 columns seen during training. Unrecognized symptom names are ignored with a logged warning rather than raising an error, so a typo in one symptom does not block a prediction based on the remaining valid ones.

---

## Models & Hyperparameters

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=2000`, `solver="lbfgs"`, `random_state=42` |
| **Random Forest** | `n_estimators=300`, `random_state=42`, `n_jobs=-1` |
| **XGBoost** | `n_estimators=300`, `learning_rate=0.1`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |

> **Compatibility note:** The original design specified `multi_class="multinomial"` for Logistic Regression. This parameter was deprecated in scikit-learn 1.5 and **removed entirely in scikit-learn ≥ 1.7** (passing it raises a `TypeError`). It has been intentionally omitted — the `"lbfgs"` solver used here natively applies multinomial (softmax) loss whenever the target has more than two classes, so equivalent behavior is preserved automatically without the deprecated parameter.

---

## Evaluation Metrics

For every model, the pipeline computes:

- **Accuracy**
- **Precision (weighted)**
- **Recall (weighted)**
- **F1-score (weighted)** — used as the model-selection criterion
- **ROC-AUC (One-vs-Rest)**
- **Full classification report** (per-class precision/recall/F1)
- **Confusion matrix** (saved as a heatmap in `plots/`)

The best model is the one with the highest weighted F1-score, automatically saved to `models/best_model.pkl`. A `models/model_metadata.json` file records the comparison metrics for all trained models, plus which one was selected and whether it requires scaled inputs.

### Handling rare disease classes

With 100 classes spread across the dataset, some diseases may have very few samples. The pipeline handles this robustly:

- If a class has fewer than 2 samples, a stratified train/test split is impossible — the pipeline detects this and automatically falls back to a non-stratified split with a logged warning, rather than crashing.
- If a class ends up entirely absent from the training fold, the affected model's `predict_proba` output naturally has fewer columns than the total class count. The evaluation code re-aligns probability columns to the full label space (using each model's `classes_` attribute) before computing ROC-AUC, so the metric remains mathematically correct rather than silently misaligned.

---

## Exploratory Data Analysis

`main.py` generates the following plots into `plots/`:

1. **Class distribution** — full frequency breakdown across all 100 diseases.
2. **Top diseases** — the most frequent diagnoses in the dataset.
3. **Symptom frequency** — the most commonly reported symptoms.
4. **Correlation heatmap** — pairwise correlation among the most frequent symptoms (subsampled for readability, since 230×230 is too dense to interpret visually).
5. **Feature sparsity** — distribution of how many symptoms are reported per patient case.

---

## Logging

All modules use Python's `logging` module (no `print()` statements in the core pipeline) and write to both the console and `logs/pipeline.log`. Each pipeline stage is clearly delimited (`STEP 1/6`, `STEP 2/6`, ...) for easy tracing.

---

## Reproducibility

`random_state=42` is used consistently across the train/test split and every model that accepts a random seed, ensuring that pipeline runs are deterministic given the same input data.

---
## Results

The pipeline trained and compared three machine learning models on 96,088 patient records containing 230 binary symptom features and 100 disease classes.

| Model               |   Accuracy | Weighted F1 Score |    ROC-AUC |
| ------------------- | ---------: | ----------------: | ---------: |
| Logistic Regression | **89.80%** |        **89.87%** | **0.9996** |
| Random Forest       |     87.00% |            87.01% |     0.9980 |
| XGBoost             |     86.99% |            87.02% |     0.9993 |

### Best Model

**Logistic Regression**

* Accuracy: **89.80%**
* Weighted F1-score: **89.87%**
* ROC-AUC (OvR): **0.9996**

The best model is automatically selected based on the highest weighted F1-score and saved for future inference.

## Future Improvements

* Hyperparameter optimization using Optuna
* Ensemble learning techniques
* Explainable AI using SHAP
* Streamlit-based web application
* Docker containerization
* REST API deployment using FastAPI
* Integration with electronic health record systems
* Disease recommendation dashboard

## Author

**Vignesh R**
2nd-Year B.Tech Artificial Intelligence & Machine Learning Student
Jain University, Bengaluru

Final Project — CodeAlpha Machine Learning Internship

GitHub: https://github.com/Vigneshrethinaswamy
