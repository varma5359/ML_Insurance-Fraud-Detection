# Insurance Fraud Detection Using Machine Learning

An end-to-end machine learning project that predicts whether an insurance claim may be fraudulent. This project demonstrates data cleaning, preprocessing, model training, evaluation, and prediction using Python and scikit-learn.

**Project status:** All five modules have passed the reported validation checks. The model is an educational baseline, not a production-ready fraud detection system.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Dataset](#3-dataset)
4. [Technology Stack and Skills](#4-technology-stack-and-skills)
5. [Project Structure](#5-project-structure)
6. [End-to-End Workflow](#6-end-to-end-workflow)
7. [Module-Wise Explanation](#7-module-wise-explanation)
8. [Model Results](#8-model-results)
9. [Installation and Setup](#9-installation-and-setup)
10. [Run the Project](#10-run-the-project)
11. [Predict a New Claim](#11-predict-a-new-claim)
12. [Output Files](#12-output-files)
13. [Key Learnings](#13-key-learnings)
14. [Limitations and Future Improvements](#14-limitations-and-future-improvements)
15. [Disclaimer](#15-disclaimer)

---

## 1. Project Overview

Insurance companies receive claims for accidents, vehicle damage, and other insured events. Some claims may contain fraudulent information, while most are legitimate. Machine learning can help prioritize claims for human review.

This project builds a **binary classification model**:

- `0` — Legitimate claim
- `1` — Fraud claim

### Project at a glance

| Item | Details |
|---|---|
| Business domain | Insurance and Claims Analytics |
| Problem type | Supervised binary classification |
| Programming language | Python |
| Data processing | pandas |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Development tools | Jupyter Notebook and Windows CMD |
| Dataset size | 1,000 rows and 39 original columns |
| Train/test split | 800 training rows and 200 test rows |
| Models compared | Logistic Regression and Random Forest |
| Selected baseline | Logistic Regression |

## 2. Business Problem

An insurer wants to identify claims that may deserve further investigation. A machine learning model can help prioritize review, but it should not make the final fraud determination.

Key business questions:

- How many fraudulent claims does the model detect?
- How many legitimate claims are incorrectly flagged?
- How many fraudulent claims are missed?
- How well does the model perform on data it did not use during training?

### Why evaluation metrics matter

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** Of the claims flagged as fraud, how many were fraud in the labeled data?
- **Recall:** Of all actual fraud claims, how many did the model detect?
- **F1-score:** Combines precision and recall into one metric.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across classification thresholds.
- **Confusion matrix:** Shows correct predictions, false positives, and false negatives.

The appropriate balance depends on investigation capacity, claim costs, and the consequences of errors.

## 3. Dataset

This project uses `insurance_claims.csv` from the public NUS Fintech Society repository.

**Dataset source:** https://github.com/NUS-Fintech-Society/ML_Insurance-Fraud-Detection

The dataset used in this project contains 1,000 rows and 39 original columns. The target column is `fraud_reported`, with original labels `Y` and `N`.

### Target distribution

| Original label | Meaning | Count |
|---|---|---:|
| `N` | Legitimate claim | 753 |
| `Y` | Fraud claim | 247 |
| **Total** | | **1,000** |

For modeling, `N` is mapped to `0` and `Y` to `1`.

### Missing values observed

| Column | Missing values |
|---|---:|
| `collision_type` | 178 |
| `authorities_contacted` | 91 |
| `property_damage` | 360 |
| `police_report_available` | 343 |

Missing feature values are handled during preprocessing. The preprocessing pipeline learns imputation values from training data rather than the entire dataset.

Review the dataset repository's license and terms before redistributing the raw data.

## 4. Technology Stack and Skills

### Technology stack

| Technology | Purpose |
|---|---|
| Python | Project implementation |
| pandas | Data loading, cleaning, and preparation |
| scikit-learn | Preprocessing, model training, and evaluation |
| Logistic Regression | Baseline classifier |
| Random Forest | Comparison classifier |
| SimpleImputer | Missing-value handling |
| StandardScaler | Numeric feature scaling |
| OneHotEncoder | Categorical feature encoding |
| ColumnTransformer | Different transformations for numeric and categorical columns |
| Pipeline | Combining preprocessing and model steps |
| GridSearchCV | Hyperparameter search using cross-validation |
| StratifiedKFold | Maintaining class proportions across validation folds |
| joblib | Saving and loading model artifacts |
| Jupyter Notebook | Interactive analysis and documentation |
| Windows CMD | Running modules and validation commands |
| Git and GitHub | Version control and portfolio publishing |

### Skills demonstrated

**Data preparation**
- Data inspection and quality checks
- Missing-value analysis
- Duplicate detection
- Text normalization
- Target validation and encoding
- Feature and target separation

**Machine learning**
- Supervised learning and binary classification
- Train/test splitting and stratification
- Numeric and categorical preprocessing
- Prevention of data leakage
- Cross-validation and hyperparameter tuning
- Model comparison and persistence

**Model evaluation**
- Accuracy, precision, recall, F1-score, and ROC-AUC
- Classification reports and confusion matrices
- False-positive and false-negative analysis
- Interpreting metrics for business stakeholders

**Software development**
- Modular Python code
- Reusable functions and function docstrings
- Command-line execution
- Input schema validation
- Saving and reloading model artifacts
- Independent validation of outputs

**Business skills**
- Translating a business problem into an ML task
- Understanding investigation trade-offs
- Communicating model performance and limitations

## 5. Project Structure

The source code is organized into two subpackages: `data_processing` and `modeling`.

```text
Insurance_Fraud_Detection/
├── data/
│   ├── insurance_claims.csv
│   ├── insurance_claims_cleaned.csv
│   ├── X_train_raw.csv
│   ├── X_test_raw.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── preprocessing_pipeline.joblib
│   ├── best_fraud_model.joblib
│   ├── training_results.csv
│   ├── evaluation_metrics.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── new_claim_template.csv
│   └── prediction_results.csv
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   └── preprocessing.py
│   └── modeling/
│       ├── __init__.py
│       ├── training.py
│       ├── evaluation.py
│       └── prediction.py
├── Insurance_Fraud_Detection.ipynb
├── app.py
├── requirements.txt
├── README.md
└── flow.bat
```

The `data/` directory contains the source dataset and generated outputs. Depending on the project's `.gitignore`, generated datasets and model artifacts may remain local rather than being committed to GitHub.

### Source file responsibilities

| File | Responsibility |
|---|---|
| `cleaning.py` | Inspect, clean, validate, and save the data |
| `preprocessing.py` | Split the data and fit feature transformations |
| `training.py` | Tune and compare classifiers |
| `evaluation.py` | Evaluate the selected model on the held-out test set |
| `prediction.py` | Load the saved model and predict new claims |

## 6. End-to-End Workflow

```text
Original Dataset
      |
      v
Module 1: Data Cleaning
      |
      v
Cleaned Dataset
      |
      v
Module 2: Preprocessing and Train/Test Split
      |
      +-----------------------+
      |                       |
      v                       v
Training Data              Test Data
      |                  (held aside)
      v                       |
Module 3: Model Training      |
      |                       |
      v                       |
Selected Model               |
      |                       |
      +-----------+-----------+
                  v
        Module 4: Evaluation
                  |
                  v
          Evaluation Reports

New Claim CSV + Saved Model
                  |
                  v
        Module 5: Prediction
                  |
                  v
        Prediction Results
```

The test set is not used for hyperparameter tuning or model selection. It is reserved for final evaluation.

## 7. Module-Wise Explanation

### Module 1 — Data Cleaning

**File:** `src/data_processing/cleaning.py`

**Purpose:** Make the source dataset consistent and validate its target column.

**Implementation:**
1. Load the source CSV.
2. Inspect dimensions, missing values, and target distribution.
3. Remove completely empty columns.
4. Strip text whitespace and convert blank text to missing values.
5. Identify and remove exact duplicate rows.
6. Validate target labels.
7. Remove rows with missing target labels, if any.
8. Save the cleaned dataset.

**Verified results:**

| Check | Result |
|---|---:|
| Original rows | 1,000 |
| Original columns | 39 |
| Exact duplicate rows | 0 |
| Rows removed for missing target | 0 |
| Legitimate claims | 753 |
| Fraud claims | 247 |

**Output:** `data/insurance_claims_cleaned.csv`

**Skills:** pandas, data quality checks, missing-value inspection, text cleaning, target validation, and file handling.

### Module 2 — Feature Preparation and Preprocessing

**File:** `src/data_processing/preprocessing.py`

**Purpose:** Separate the target, split the data, and prepare features for machine learning.

**Implementation:**
1. Load the cleaned dataset.
2. Separate `fraud_reported` from the input features.
3. Map `N` to `0` and `Y` to `1`.
4. Exclude selected identifiers and raw date, location, or investigation-related fields from the baseline.
5. Create an 80/20 stratified train/test split using `random_state=42`.
6. Impute numeric missing values using the median.
7. Scale numeric features using `StandardScaler`.
8. Impute categorical missing values using the most frequent value.
9. Encode categorical features using `OneHotEncoder`.
10. Fit preprocessing on training data only and transform the test data using the fitted transformations.
11. Save the splits, processed features, labels, and preprocessing artifact.

**Verified results:**

| Item | Result |
|---|---:|
| Training rows | 800 |
| Test rows | 200 |
| Selected raw features | 30 |
| Processed features | 147 |
| Training fraud rate | 24.8% |
| Test fraud rate | 24.5% |

One-hot encoding expands categorical columns, so the 30 raw input features become 147 processed features.

**Outputs:**
- `data/X_train_raw.csv`
- `data/X_test_raw.csv`
- `data/y_train.csv`
- `data/y_test.csv`
- `data/X_train_processed.csv`
- `data/X_test_processed.csv`
- `data/preprocessing_pipeline.joblib`

**Skills:** feature engineering, target encoding, stratified splitting, imputation, scaling, one-hot encoding, pipelines, and leakage prevention.

### Module 3 — Model Training and Comparison

**File:** `src/modeling/training.py`

**Purpose:** Compare classifiers and select a baseline model using mean cross-validation F1-score.

**Models compared:**
- **Logistic Regression:** A baseline binary classifier.
- **Random Forest:** An ensemble model that can learn nonlinear relationships.

**Training strategy:**
- Five-fold `StratifiedKFold` cross-validation.
- `GridSearchCV` for hyperparameter search.
- F1-score for model selection.
- Preprocessing included within the model pipeline to prevent fold-level leakage.
- Save the selected model and comparison results.
- Reload the saved model and validate predictions.

**Actual cross-validation results:**

| Model | Mean CV F1-score | Best settings |
|---|---:|---|
| Logistic Regression | **0.4593** | `C=0.1`, `class_weight='balanced'` |
| Random Forest | 0.3337 | `max_depth=10`, `n_estimators=100`, `class_weight='balanced'` |

Logistic Regression was selected because it achieved the higher mean cross-validation F1-score among the tested configurations.

**Outputs:**
- `data/best_fraud_model.joblib`
- `data/training_results.csv`

**Skills:** classification, cross-validation, hyperparameter tuning, model comparison, F1-score, pipelines, and model persistence.

### Module 4 — Model Evaluation

**File:** `src/modeling/evaluation.py`

**Purpose:** Evaluate the selected model on the reserved test set.

**Actual test results:**

| Metric | Result |
|---|---:|
| Accuracy | 0.6200 |
| Fraud precision | 0.3247 |
| Fraud recall | 0.5102 |
| Fraud F1-score | 0.3968 |
| ROC-AUC | 0.6133 |

**Confusion matrix**

Rows represent actual classes; columns represent predicted classes.

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 99 | 52 |
| Actual fraud | 24 | 25 |

**Business interpretation:**
- The model detected 25 of the 49 actual fraud cases.
- It missed 24 fraud cases.
- It incorrectly flagged 52 legitimate claims as fraud.
- Fraud precision of 32.47% means roughly one in three claims flagged as fraud was actually fraud in this test set.
- Fraud recall of 51.02% means the model detected about half of the fraud cases in this test set.

The baseline has significant room for improvement. It should not be used alone to reject claims or accuse anyone of fraud.

**Outputs:**
- `data/evaluation_metrics.csv`
- `data/classification_report.csv`
- `data/confusion_matrix.csv`

**Skills:** metrics, confusion matrix interpretation, error analysis, held-out evaluation, and business communication.

### Module 5 — Prediction and Integration

**File:** `src/modeling/prediction.py`

**Purpose:** Use the saved model pipeline to score new claim records without retraining.

**Implementation:**
1. Load the saved model.
2. Create a template with the expected feature columns.
3. Read a new claim CSV.
4. Validate the input schema, including missing and unexpected columns.
5. Reorder columns to match the training schema.
6. Predict the class and fraud probability.
7. Save the results.

**Prediction fields:**

| Field | Meaning |
|---|---|
| `predicted_class` | `0` = legitimate; `1` = fraud |
| `prediction` | Human-readable predicted label |
| `fraud_probability` | Estimated probability of the fraud class |

**Verified sample prediction:**

| Predicted class | Prediction | Fraud probability |
|---:|---|---:|
| 0 | Legitimate | 0.251737 |

The sample received an estimated fraud probability of approximately **25.17%** and was classified as legitimate at the default 0.5 threshold. This is a model estimate, not proof of the claim's true status.

**Outputs:**
- `data/new_claim_template.csv`
- `data/prediction_results.csv`

**Skills:** inference, model loading, schema validation, probability prediction, reusable pipelines, and CSV output.

## 8. Model Results

| Evaluation stage | Result |
|---|---|
| Model-selection metric | Mean five-fold cross-validation F1-score |
| Best cross-validation F1 | 0.4593 |
| Held-out test F1 | 0.3968 |
| Held-out test ROC-AUC | 0.6133 |
| Held-out fraud recall | 0.5102 |
| Prediction workflow | Independent validation passed |

Cross-validation and test F1-scores are calculated on different data partitions. Their difference alone does not establish the cause of the performance gap.

The model detects some fraud cases, but false positives and false negatives remain high. The results demonstrate an end-to-end ML workflow and provide a baseline for improvement; they do not establish high predictive performance.

## 9. Installation and Setup

### Prerequisites

- Python compatible with the dependencies in `requirements.txt`
- Windows CMD or PowerShell
- Internet access to download the dataset if needed
- Git, if publishing to GitHub

### Step 1 — Open the project directory

```cmd
cd /d D:\AI\ML\Insurance_fraud_detection
```

### Step 2 — Create and activate a virtual environment

```cmd
py -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```cmd
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Expected dependencies include pandas, scikit-learn, joblib, and notebook. Keep `requirements.txt` aligned with the versions actually tested in your environment.

### Step 4 — Download the dataset if it is missing

```cmd
mkdir data
curl -L "https://raw.githubusercontent.com/NUS-Fintech-Society/ML_Insurance-Fraud-Detection/main/insurance_claims.csv" -o "data\insurance_claims.csv"
```

If `data` already exists, CMD may report that the directory exists; continue if the dataset is present.

## 10. Run the Project

Run the modules from the project root in this order.

### Module 1 — Clean the data

```cmd
py -m py_compile src\data_processing\cleaning.py
py -m src.data_processing.cleaning
```

### Module 2 — Preprocess the data

```cmd
py -m py_compile src\data_processing\preprocessing.py
py -m src.data_processing.preprocessing
```

### Module 3 — Train and compare models

```cmd
py -m py_compile src\modeling\training.py
py -m src.modeling.training
```

### Module 4 — Evaluate the selected model

```cmd
py -m py_compile src\modeling\evaluation.py
py -m src.modeling.evaluation
```

### Module 5 — Prepare the prediction template

```cmd
py -m py_compile src\modeling\prediction.py
py -m src.modeling.prediction
```

Run the modules in order because later modules depend on artifacts produced by earlier modules.

### Run the Jupyter Notebook

```cmd
py -m notebook
```

Open `Insurance_Fraud_Detection.ipynb` and execute its cells in order. The notebook supports interactive learning, while the Python modules provide reusable implementations.

## 11. Predict a New Claim

After Module 5 creates the input template, copy it:

```cmd
copy data\new_claim_template.csv data\new_claim.csv
notepad data\new_claim.csv
```

Edit the values while preserving the expected column names and CSV structure. Include all expected feature columns and do not add the target column `fraud_reported`.

Run prediction:

```cmd
py -m src.modeling.prediction --input data\new_claim.csv
```

The default output is:

```text
data\prediction_results.csv
```

To select a different output file:

```cmd
py -m src.modeling.prediction --input data\new_claim.csv --output data\my_predictions.csv
```

The template starts from a sample training row. Replace its values before interpreting the output as a hypothetical claim. Predictions depend on input quality and model limitations.

## 12. Output Files

| File | Purpose |
|---|---|
| `insurance_claims.csv` | Original dataset |
| `insurance_claims_cleaned.csv` | Cleaned dataset |
| `X_train_raw.csv`, `X_test_raw.csv` | Raw feature splits |
| `y_train.csv`, `y_test.csv` | Target splits |
| `X_train_processed.csv`, `X_test_processed.csv` | Transformed feature splits |
| `preprocessing_pipeline.joblib` | Fitted preprocessing artifact |
| `best_fraud_model.joblib` | Selected model pipeline |
| `training_results.csv` | Cross-validation model comparison |
| `evaluation_metrics.csv` | Test metrics |
| `classification_report.csv` | Per-class evaluation report |
| `confusion_matrix.csv` | Actual-versus-predicted counts |
| `new_claim_template.csv` | New-claim input template |
| `prediction_results.csv` | Prediction results |

These files are generated by running the modules. Depending on `.gitignore`, generated data and model artifacts may remain local instead of being committed to GitHub.

## 13. Key Learnings

After completing the project, a learner should be able to:

1. Translate a business problem into a supervised classification task.
2. Inspect and clean a real-world dataset.
3. Separate input features from the target.
4. Prepare numeric and categorical data.
5. Avoid preprocessing leakage by fitting transformations on training data only.
6. Use stratified splitting and cross-validation.
7. Compare models using an appropriate selection metric.
8. Explain why accuracy alone can be misleading.
9. Interpret precision, recall, F1-score, ROC-AUC, and a confusion matrix.
10. Save and reuse a model pipeline for new predictions.
11. Organize code into focused Python modules.
12. Communicate model performance, limitations, and business risks.

## 14. Limitations and Future Improvements

### Model improvement
- Compare additional classifiers, such as Gradient Boosting, with the baseline.
- Investigate class imbalance and perform any resampling only within training folds.
- Study precision-recall curves and tune the decision threshold using validation data and business costs.
- Assess probability calibration before using scores to prioritize investigations.

### Data quality and leakage
- Review feature definitions and when each feature becomes available.
- Check that no feature reveals information only known after investigation or claim outcome.
- Investigate duplicate or related claims across data partitions.
- Validate on newer or external data to measure generalization.
- Confirm the dataset's provenance, representativeness, and permitted use.

### Engineering
- Pin tested dependency versions.
- Add checks for input schemas, artifacts, and output validity.
- Document data download and reproducibility steps.
- Consider a user interface only after assessing model suitability.
- Add monitoring, security, auditability, and retraining controls before operational deployment.

### Fairness, privacy, and human review
Insurance decisions can affect people's finances and access to services. Assess model errors and potential bias across relevant groups where lawful and appropriate. Protect personal information and require qualified human review. A model score is not proof of fraud and should not be used alone to reject a claim.

## 15. Disclaimer

This project is for educational and portfolio purposes only. It does not provide insurance, legal, or fraud-investigation advice. Predictions are statistical estimates from a limited dataset and must not be treated as definitive fraud determinations.

---

## Author

**Ravi Varma**  
Data Science • Machine Learning • AI Engineering
