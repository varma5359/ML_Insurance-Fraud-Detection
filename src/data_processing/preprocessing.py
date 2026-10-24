
"""Prepare insurance claims for leakage-aware machine learning."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "insurance_claims_cleaned.csv"
PIPELINE_PATH = DATA_DIR / "preprocessing_pipeline.joblib"

TARGET_COLUMN = "fraud_reported"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Exclude identifiers, raw date strings, high-cardinality location data,
# and fields that may only be known after claim investigation.
EXCLUDED_COLUMNS = [
    "policy_number",
    "insured_zip",
    "incident_location",
    "policy_bind_date",
    "incident_date",
    "authorities_contacted",
    "police_report_available",
    "incident_severity",
]


def load_cleaned_data(file_path):
    """Load the cleaned claims CSV and verify that the fraud target exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {file_path}. "
            "Run Module 1 first."
        )

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("The cleaned dataset is empty.")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' was not found."
        )

    return data


def prepare_features_and_target(data):
    """Separate the target, validate labels, and remove excluded features."""
    data = data.copy()

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("string").str.strip().str.upper()

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The fraud target contains missing values.")

    invalid_labels = set(data[TARGET_COLUMN].unique()) - {"Y", "N"}

    if invalid_labels:
        raise ValueError(f"Unexpected fraud labels: {invalid_labels}")

    # Encode the target as 0 = legitimate and 1 = fraud-reported.
    y = data[TARGET_COLUMN].map({"N": 0, "Y": 1}).astype("int64")

    # Keep the target separate so it cannot enter the feature pipeline.
    X = data.drop(columns=[TARGET_COLUMN])

    excluded_present = [
        column for column in EXCLUDED_COLUMNS if column in X.columns
    ]
    X = X.drop(columns=excluded_present)

    if X.empty:
        raise ValueError("No input features remain after exclusions.")

    # Convert non-finite numeric values to missing values.
    numeric_columns = X.select_dtypes(include=["number"]).columns
    X[numeric_columns] = X[numeric_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    # Standardize pandas string/object columns before encoding.
    text_columns = X.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        X[column] = X[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        X[column] = X[column].replace(r"^\s*$", pd.NA, regex=True)

    print("\n--- Feature preparation ---")
    print("Excluded columns:", excluded_present)
    print("Feature count:", X.shape[1])
    print("Target counts:")
    print(y.value_counts().sort_index().rename(index={0: "Legitimate (0)", 1: "Fraud (1)"}))

    return X, y


def build_preprocessing_pipeline(X_train):
    """Build imputers, one-hot encoding, and scaling for training features."""
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in X_train.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessing


def save_split_data(X_train, X_test, y_train, y_test):
    """Save untransformed training and test splits for inspection."""
    X_train.to_csv(DATA_DIR / "X_train_raw.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test_raw.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(DATA_DIR / "y_test.csv", index=False)


def main():
    """Split claims, fit preprocessing on training data, and validate outputs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_cleaned_data(INPUT_PATH)
    X, y = prepare_features_and_target(data)

    # Stratification preserves the fraud/non-fraud proportions in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Critical leakage prevention: fit preprocessing on training data only.
    preprocessing = build_preprocessing_pipeline(X_train)
    X_train_processed = preprocessing.fit_transform(X_train)
    X_test_processed = preprocessing.transform(X_test)

    feature_names = preprocessing.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed, columns=feature_names
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names
    )

    # Store processed features separately from target labels.
    X_train_processed.to_csv(DATA_DIR / "X_train_processed.csv", index=False)
    X_test_processed.to_csv(DATA_DIR / "X_test_processed.csv", index=False)

    save_split_data(X_train, X_test, y_train, y_test)
    joblib.dump(preprocessing, PIPELINE_PATH)

    # Basic validation of the generated modeling datasets.
    assert len(X_train) + len(X_test) == len(data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert list(X_train_processed.columns) == list(X_test_processed.columns)
    assert X_train_processed.shape[1] > 0
    assert not X_train_processed.isna().any().any()
    assert not X_test_processed.isna().any().any()
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns

    print("\n--- Module 2 summary ---")
    print(f"Original rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Original input features: {X.shape[1]}")
    print(f"Encoded features: {X_train_processed.shape[1]}")
    print(f"Training fraud rate: {y_train.mean():.3f}")
    print(f"Test fraud rate: {y_test.mean():.3f}")
    print(f"Saved preprocessing pipeline: {PIPELINE_PATH}")

    print("\nPASS: Module 2 basic validation completed.")


if __name__ == "__main__":
    main()
