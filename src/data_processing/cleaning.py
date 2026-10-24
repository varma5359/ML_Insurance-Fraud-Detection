
"""Load, inspect, clean, and validate insurance claim data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "insurance_claims.csv"
CLEANED_DATA_PATH = DATA_DIR / "insurance_claims_cleaned.csv"
TARGET_COLUMN = "fraud_reported"


def load_data(file_path):
    """Load a CSV file and interpret common missing-value markers."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(
        file_path,
        na_values=["?", "NA", "N/A", "na", "n/a"],
        keep_default_na=True,
    )


def inspect_data(data):
    """Print the dataset dimensions, data types, missing values, and target distribution."""
    print("\n--- Dataset overview ---")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print("\nColumn data types:")
    print(data.dtypes.to_string())

    print("\nMissing values by column:")
    missing = data.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "No missing values.")

    print("\nExact duplicate rows:", data.duplicated().sum())

    if TARGET_COLUMN in data.columns:
        print("\nFraud target distribution:")
        print(data[TARGET_COLUMN].value_counts(dropna=False).to_string())


def clean_data(data):
    """Normalize missing markers, remove exact duplicates, and validate fraud labels."""
    cleaned = data.copy()

    # Remove columns that contain no usable values.
    empty_columns = cleaned.columns[cleaned.isna().all()].tolist()
    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)

    # Remove unused CSV index columns only when they are entirely empty.
    empty_index_columns = [
        column
        for column in cleaned.columns
        if column.lower().startswith("unnamed") and cleaned[column].isna().all()
    ]
    if empty_index_columns:
        cleaned = cleaned.drop(columns=empty_index_columns)

    # Remove leading and trailing spaces from text values.
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Convert whitespace-only strings into missing values.
    for column in text_columns:
        cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing. "
            f"Available columns: {cleaned.columns.tolist()}"
        )

    # Standardize the target labels.
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(
        lambda value: value.upper() if isinstance(value, str) else value
    )

    # Rows without a known target cannot be used as labeled training examples.
    missing_target_count = int(cleaned[TARGET_COLUMN].isna().sum())
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()

    valid_labels = {"Y", "N"}
    actual_labels = set(cleaned[TARGET_COLUMN].unique())
    invalid_labels = actual_labels - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Unexpected fraud labels: {sorted(invalid_labels)}. "
            f"Expected only {sorted(valid_labels)}."
        )

    print("\n--- Cleaning summary ---")
    print(f"Exact duplicate rows removed: {duplicate_count}")
    print(f"Rows removed for missing fraud labels: {missing_target_count}")
    print(f"Rows remaining: {len(cleaned)}")
    print(f"Columns remaining: {len(cleaned.columns)}")
    print("\nMissing values remaining by column:")
    remaining_missing = cleaned.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    print(
        remaining_missing.to_string()
        if not remaining_missing.empty
        else "No missing values."
    )

    print("\nValidated fraud label counts:")
    print(cleaned[TARGET_COLUMN].value_counts().to_string())

    return cleaned


def save_cleaned_data(data, file_path):
    """Save the cleaned dataset as a CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)
    print(f"\nCleaned dataset saved to: {file_path}")


def main():
    """Run the Module 1 data inspection and cleaning workflow."""
    raw_data = load_data(RAW_DATA_PATH)

    inspect_data(raw_data)

    cleaned_data = clean_data(raw_data)
    save_cleaned_data(cleaned_data, CLEANED_DATA_PATH)

    # Verify the saved CSV can be read back successfully.
    saved_data = load_data(CLEANED_DATA_PATH)

    if saved_data.empty:
        raise ValueError("Validation failed: the cleaned dataset is empty.")

    if TARGET_COLUMN not in saved_data.columns:
        raise ValueError("Validation failed: the fraud target is missing.")

    if not set(saved_data[TARGET_COLUMN].unique()).issubset({"Y", "N"}):
        raise ValueError("Validation failed: unexpected target labels.")

    print("\nPASS: Module 1 basic validation completed.")
    print(f"Saved dataset shape: {saved_data.shape}")


if __name__ == "__main__":
    main()
