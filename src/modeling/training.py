
"""Train and compare leakage-aware insurance fraud classifiers."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_processing.preprocessing import (
    TARGET_COLUMN,
    build_preprocessing_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

X_TRAIN_PATH = DATA_DIR / "X_train_raw.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"

MODEL_PATH = DATA_DIR / "best_fraud_model.joblib"
RESULTS_PATH = DATA_DIR / "training_results.csv"


def load_training_data():
    """Load training features and labels created by Module 2."""
    if not X_TRAIN_PATH.exists() or not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Training data not found. Run Module 2 first."
        )

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET_COLUMN]

    if X_train.empty:
        raise ValueError("Training features are empty.")

    if len(X_train) != len(y_train):
        raise ValueError("Training features and labels have different row counts.")

    if TARGET_COLUMN in X_train.columns:
        raise ValueError("The target column must not appear in training features.")

    if y_train.isna().any() or not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must contain only 0 and 1.")

    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    return X_train, y_train


def build_model_pipeline(X_train, classifier):
    """Combine training-fitted preprocessing with a classifier."""
    preprocessing = build_preprocessing_pipeline(X_train)

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def train_and_compare_models(X_train, y_train):
    """Tune two classifiers using stratified cross-validation on training data."""
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model_configurations = [
        {
            "name": "Logistic Regression",
            "classifier": LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
            "parameters": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "Random Forest",
            "classifier": RandomForestClassifier(
                random_state=42,
                n_jobs=1,
            ),
            "parameters": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    ]

    results = []
    best_model = None
    best_score = float("-inf")
    best_name = None

    for configuration in model_configurations:
        print(f"\nTraining: {configuration['name']}")

        pipeline = build_model_pipeline(
            X_train,
            configuration["classifier"],
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration["parameters"],
            scoring="f1",
            cv=cross_validation,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        score = float(search.best_score_)

        results.append(
            {
                "model": configuration["name"],
                "cv_f1": score,
                "best_parameters": str(search.best_params_),
            }
        )

        print(f"Best cross-validation F1: {score:.4f}")
        print("Best parameters:", search.best_params_)

        if score > best_score:
            best_score = score
            best_model = search.best_estimator_
            best_name = configuration["name"]

    return best_model, best_name, best_score, pd.DataFrame(results)


def main():
    """Train, compare, and save the best cross-validated fraud classifier."""
    X_train, y_train = load_training_data()

    best_model, best_name, best_score, results = train_and_compare_models(
        X_train,
        y_train,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    results.to_csv(RESULTS_PATH, index=False)

    # Verify the saved artifact can be reloaded and predicts training rows.
    loaded_model = joblib.load(MODEL_PATH)
    predictions = loaded_model.predict(X_train.head(5))

    if len(predictions) != min(5, len(X_train)):
        raise ValueError("Reloaded model returned an unexpected prediction count.")

    if not set(predictions).issubset({0, 1}):
        raise ValueError("Model returned unexpected class labels.")

    if results.empty or results["cv_f1"].isna().any():
        raise ValueError("Training results are empty or contain invalid scores.")

    print("\n--- Module 3 summary ---")
    print(results.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Best mean cross-validation F1: {best_score:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved comparison: {RESULTS_PATH}")
    print("Sample predictions:", predictions.tolist())
    print("\nPASS: Module 3 basic validation completed.")


if __name__ == "__main__":
    main()
