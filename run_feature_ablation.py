
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_UHI_Spatial_Split_2021_2025.csv"
TARGET = "LST_C"

FEATURE_SETS = {
    "Full model": [
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
        "year",
        "doy_sin",
        "doy_cos",
    ],

    "Without day-of-year": [
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
        "year",
    ],

    "Environmental only": [
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
    ],
}


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def make_model():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=5,
        max_features=1.0,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )


def calculate_metrics(actual, predicted):
    return {
        "samples": len(actual),
        "RMSE_C": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAE_C": mean_absolute_error(
            actual, predicted
        ),
        "R2": r2_score(actual, predicted),
        "mean_error_C": np.mean(
            np.asarray(actual) - np.asarray(predicted)
        ),
    }


# ==================================================
# 3. LOAD AND VALIDATE
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    all_features = sorted({
        feature
        for features in FEATURE_SETS.values()
        for feature in features
    })

    required = all_features + [TARGET, "split"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for column in all_features + [TARGET]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    if data[required].isna().any().any():
        raise ValueError(
            "Missing or invalid values found. "
            "Audit the input data before continuing."
        )

    data["split"] = (
        data["split"].astype(str).str.strip().str.lower()
    )

    train = data[data["split"] == "train"].copy()
    validation = data[
        data["split"] == "validation"
    ].copy()
    test = data[data["split"] == "test"].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("One or more data partitions are empty.")

    print("FEATURE ABLATION STUDY")
    print("-" * 60)
    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")

    # ==================================================
    # 4. VALIDATION AND HELD-OUT TEST EVALUATION
    # ==================================================

    validation_results = []
    test_results = []
    importance_rows = []
    prediction_frames = []

    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    for model_name, features in FEATURE_SETS.items():

        print(f"\nTraining: {model_name}")

        # Train on training data, evaluate on validation.
        validation_model = make_model()

        validation_model.fit(
            train[features],
            train[TARGET],
        )

        validation_predictions = (
            validation_model.predict(validation[features])
        )

        validation_metrics = calculate_metrics(
            validation[TARGET],
            validation_predictions,
        )

        validation_results.append({
            "model": model_name,
            "features": ", ".join(features),
            **validation_metrics,
        })

        # Retrain the same configuration on training +
        # validation. Keep the test set untouched until now.
        final_model = make_model()

        final_model.fit(
            train_validation[features],
            train_validation[TARGET],
        )

        test_predictions = final_model.predict(
            test[features]
        )

        test_metrics = calculate_metrics(
            test[TARGET],
            test_predictions,
        )

        test_results.append({
            "model": model_name,
            "feature_count": len(features),
            **test_metrics,
        })

        # Record feature importance for interpretation.
        for feature, importance in zip(
            features, final_model.feature_importances_
        ):
            importance_rows.append({
                "model": model_name,
                "feature": feature,
                "importance": importance,
            })

        # Save test predictions for each model.
        metadata_columns = [
            column for column in [
                "date", "year", "longitude", "latitude",
                "block_x", "block_y"
            ]
            if column in test.columns
        ]

        predictions_frame = test[
            metadata_columns
        ].copy()

        predictions_frame["model"] = model_name
        predictions_frame["LST_C_observed"] = (
            test[TARGET].to_numpy()
        )
        predictions_frame["LST_C_predicted"] = (
            test_predictions
        )
        predictions_frame["residual_C"] = (
            test[TARGET].to_numpy() - test_predictions
        )

        prediction_frames.append(predictions_frame)

    # ==================================================
    # 5. PRINT AND SAVE RESULTS
    # ==================================================

    validation_df = pd.DataFrame(validation_results)
    test_df = pd.DataFrame(test_results)

    importance_df = pd.DataFrame(importance_rows)

    validation_df = validation_df.sort_values(
        "RMSE_C"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        "RMSE_C"
    ).reset_index(drop=True)

    print("\nVALIDATION RESULTS")
    print("-" * 60)
    print(validation_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 60)
    print(test_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    validation_df.to_csv(
        FOLDER / "Bengaluru_UHI_Ablation_Validation.csv",
        index=False,
    )

    test_df.to_csv(
        FOLDER / "Bengaluru_UHI_Ablation_Test.csv",
        index=False,
    )

    importance_df.to_csv(
        FOLDER / "Bengaluru_UHI_Ablation_Importances.csv",
        index=False,
    )

    pd.concat(
        prediction_frames, ignore_index=True
    ).to_csv(
        FOLDER / "Bengaluru_UHI_Ablation_Predictions.csv",
        index=False,
    )

    print("\nFILES CREATED")
    print("-" * 60)
    print("Bengaluru_UHI_Ablation_Validation.csv")
    print("Bengaluru_UHI_Ablation_Test.csv")
    print("Bengaluru_UHI_Ablation_Importances.csv")
    print("Bengaluru_UHI_Ablation_Predictions.csv")

    print("\nFeature ablation study completed.")


if __name__ == "__main__":
    main()
