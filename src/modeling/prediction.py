
"""Generate fraud predictions for new insurance claims."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = DATA_DIR / "best_fraud_model.joblib"
TRAINING_FEATURES_PATH = DATA_DIR / "X_train_raw.csv"
TEMPLATE_PATH = DATA_DIR / "new_claim_template.csv"
DEFAULT_INPUT_PATH = DATA_DIR / "new_claim.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "prediction_results.csv"


def load_model():
    """Load the saved preprocessing and classification pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Complete Module 3 first."
        )

    return joblib.load(MODEL_PATH)


def create_claim_template():
    """Create an editable input template using the training feature schema."""
    if not TRAINING_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Training features not found. Complete Module 2 first."
        )

    training_features = pd.read_csv(TRAINING_FEATURES_PATH)

    if training_features.empty:
        raise ValueError("Training features are empty.")

    template = training_features.head(1).copy()
    template.to_csv(TEMPLATE_PATH, index=False)

    return template


def predict_claim(model, claim_data):
    """Predict fraud labels and probabilities for supplied claim features."""
    if not isinstance(claim_data, pd.DataFrame) or claim_data.empty:
        raise ValueError("Claim data must be a non-empty pandas DataFrame.")

    if claim_data.columns.duplicated().any():
        raise ValueError("Claim data contains duplicate column names.")

    expected_columns = list(model.feature_names_in_)
    supplied_columns = list(claim_data.columns)

    missing_columns = sorted(set(expected_columns) - set(supplied_columns))
    extra_columns = sorted(set(supplied_columns) - set(expected_columns))

    if missing_columns or extra_columns:
        raise ValueError(
            f"Input feature mismatch. Missing columns: {missing_columns}. "
            f"Unexpected columns: {extra_columns}."
        )

    # Match the exact feature order used during training.
    claim_data = claim_data[expected_columns].copy()

    predictions = model.predict(claim_data)
    fraud_probabilities = model.predict_proba(claim_data)[:, 1]

    results = pd.DataFrame(
        {
            "predicted_class": predictions.astype(int),
            "prediction": [
                "Fraud" if label == 1 else "Legitimate"
                for label in predictions
            ],
            "fraud_probability": fraud_probabilities,
        }
    )

    return results


def predict_from_csv(input_path, output_path):
    """Load claims from CSV, predict them, and save the results."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            f"Copy {TEMPLATE_PATH.name} to {DEFAULT_INPUT_PATH.name} "
            "and edit the claim values first."
        )

    claim_data = pd.read_csv(input_path)
    model = load_model()
    results = predict_claim(model, claim_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_results = pd.concat(
        [claim_data.reset_index(drop=True), results],
        axis=1,
    )
    combined_results.to_csv(output_path, index=False)

    print("--- Prediction results ---")
    print(results.to_string(index=False))
    print(f"\nSaved prediction file: {output_path}")
    print(
        "\nNote: Predictions are model estimates, not proof "
        "that a claim is fraudulent."
    )

    return results


def main():
    """Create an input template or predict claims from a supplied CSV."""
    parser = argparse.ArgumentParser(
        description="Predict insurance claim fraud using the saved model."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a CSV containing new claim features.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for saving prediction results.",
    )
    args = parser.parse_args()

    if args.input:
        predict_from_csv(args.input, args.output)
    else:
        create_claim_template()
        print(f"Claim template created: {TEMPLATE_PATH}")
        print("\nNext steps:")
        print(
            f'1. Run: copy "{TEMPLATE_PATH}" "{DEFAULT_INPUT_PATH}"'
        )
        print(f"2. Edit {DEFAULT_INPUT_PATH.name} with the claim details.")
        print(
            "3. Run: py -m src.modeling.prediction "
            "--input data\\new_claim.csv"
        )


if __name__ == "__main__":
    main()
