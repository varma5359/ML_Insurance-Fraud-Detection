
"""Evaluate the saved insurance fraud model on the untouched test set."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = DATA_DIR / "best_fraud_model.joblib"
X_TEST_PATH = DATA_DIR / "X_test_raw.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

METRICS_PATH = DATA_DIR / "evaluation_metrics.csv"
REPORT_PATH = DATA_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = DATA_DIR / "confusion_matrix.csv"

TARGET_COLUMN = "fraud_reported"


def load_test_data():
    """Load the reserved test features and labels."""
    required_files = [MODEL_PATH, X_TEST_PATH, Y_TEST_PATH]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}. "
                "Complete Modules 2 and 3 first."
            )

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET_COLUMN]

    if X_test.empty or len(X_test) != len(y_test):
        raise ValueError("Test features and labels are empty or misaligned.")

    if TARGET_COLUMN in X_test.columns:
        raise ValueError("The target column must not appear in test features.")

    if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Test labels must contain only 0 and 1.")

    if y_test.nunique() != 2:
        raise ValueError("Test data must contain both target classes.")

    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics without fitting on test data."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return metrics, pd.DataFrame(report).transpose(), matrix


def save_evaluation_results(metrics, report, matrix):
    """Save test metrics, classification report and confusion matrix."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items()]
    ).to_csv(METRICS_PATH, index=False)

    report.to_csv(REPORT_PATH, index=True)

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Legitimate (0)", "Actual Fraud (1)"],
        columns=["Predicted Legitimate (0)", "Predicted Fraud (1)"],
    )
    matrix_df.to_csv(CONFUSION_MATRIX_PATH, index=True)


def main():
    """Evaluate the saved model and validate the output files."""
    X_test, y_test = load_test_data()
    model = joblib.load(MODEL_PATH)

    metrics, report, matrix = evaluate_model(model, X_test, y_test)
    save_evaluation_results(metrics, report, matrix)

    print("--- Test set evaluation ---")
    print(f"Test rows: {len(X_test)}")

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")

    print("\nClassification report:")
    print(report.round(4).to_string())

    print("\nConfusion matrix:")
    print(matrix)

    print("\nSaved files:")
    print(METRICS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_MATRIX_PATH)

    print("\nPASS: Module 4 basic validation completed.")


if __name__ == "__main__":
    main()
