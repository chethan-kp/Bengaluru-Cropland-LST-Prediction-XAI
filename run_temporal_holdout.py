
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Deduplicated_2021_2025.csv"
)

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

MODEL_NAMES = ["Linear Regression", "Random Forest"]


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def create_model(name):

    if name == "Linear Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            max_features=1.0,
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "samples": int(len(actual)),
        "RMSE_C": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "MAE_C": float(
            mean_absolute_error(actual, predicted)
        ),
        "R2": float(r2_score(actual, predicted)),
        "mean_error_C": float(
            np.mean(actual - predicted)
        ),
    }


# ==================================================
# 3. LOAD AND VALIDATE DATA
# ==================================================

def main():

    print("TEMPORAL HOLDOUT EXPERIMENT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE.name}"
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "date",
        "product_id",
        "LST_C",
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    if data["date"].isna().any():
        raise ValueError("Invalid acquisition dates found.")

    # Derive year and seasonal features from acquisition date.
    data["year"] = data["date"].dt.year

    day = data["date"].dt.dayofyear
    angle = 2 * np.pi * day / 365.25

    data["doy_sin"] = np.sin(angle)
    data["doy_cos"] = np.cos(angle)

    all_features = sorted({
        feature
        for features in FEATURE_SETS.values()
        for feature in features
    })

    for column in all_features + ["LST_C"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    numeric = data[all_features + ["LST_C"]]

    if numeric.isna().any().any():
        raise ValueError(
            "Missing or non-numeric model values found."
        )

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Infinite model values found.")

    available_years = set(data["year"].unique())

    required_years = {2021, 2022, 2023, 2024, 2025}

    if not required_years.issubset(available_years):
        raise ValueError(
            f"Expected data for 2021–2025; found "
            f"{sorted(available_years)}"
        )

    # Strict chronological partition:
    # Train      = 2021–2023
    # Validation = 2024
    # Test       = 2025

    data["temporal_split"] = np.select(
        [
            data["year"].between(2021, 2023),
            data["year"] == 2024,
            data["year"] == 2025,
        ],
        [
            "train",
            "validation",
            "test",
        ],
        default="exclude",
    )

    data = data[data["temporal_split"] != "exclude"].copy()

    # Ensure dates and complete scenes remain in one partition.
    for column in ["date", "product_id"]:

        crossing = (
            data.groupby(column)["temporal_split"].nunique()
        )

        violations = int((crossing > 1).sum())

        print(f"{column} values crossing partitions: {violations}")

        if violations:
            raise ValueError(
                f"Temporal leakage detected in {column}."
            )

    # Save the temporal split separately.
    split_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Split_2021_2025.csv"
    )

    data.to_csv(split_file, index=False)

    print(f"\nRows in temporal split: {len(data):,}")
    print("\nROWS BY YEAR AND PARTITION")
    print("-" * 65)

    print(
        pd.crosstab(
            data["year"],
            data["temporal_split"],
        ).to_string()
    )

    print("\nSCENES BY PARTITION")
    print("-" * 65)

    print(
        data.groupby("temporal_split")
        .agg(
            rows=("temporal_split", "size"),
            scenes=("product_id", "nunique"),
            years=("year", "nunique"),
        )
        .to_string()
    )

    train = data[data["temporal_split"] == "train"].copy()

    validation = data[
        data["temporal_split"] == "validation"
    ].copy()

    test = data[data["temporal_split"] == "test"].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("A temporal partition is empty.")


    # ==================================================
    # 4. VALIDATION: COMPARE ALL SIX COMBINATIONS
    # ==================================================

    validation_results = []

    for feature_set, features in FEATURE_SETS.items():

        for model_name in MODEL_NAMES:

            print(
                f"\nValidation: {feature_set} | {model_name}"
            )

            model = create_model(model_name)

            model.fit(
                train[features],
                train["LST_C"],
            )

            predictions = model.predict(
                validation[features]
            )

            validation_results.append({
                "feature_set": feature_set,
                "model": model_name,
                "feature_count": len(features),
                **calculate_metrics(
                    validation["LST_C"], predictions
                ),
            })

    validation_df = pd.DataFrame(
        validation_results
    ).sort_values("RMSE_C").reset_index(drop=True)

    print("\nVALIDATION RESULTS — 2024")
    print("-" * 65)

    print(validation_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    # Select the configuration using validation data only.
    best = validation_df.iloc[0]

    best_feature_set = best["feature_set"]
    best_model_name = best["model"]

    print("\nSELECTED CONFIGURATION")
    print("-" * 65)
    print(f"Feature set: {best_feature_set}")
    print(f"Model:       {best_model_name}")
    print(f"Validation RMSE: {best['RMSE_C']:.4f} °C")


    # ==================================================
    # 5. TEST ON UNSEEN YEAR 2025
    # ==================================================

    # Refit using all data through 2024.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    test_results = []
    prediction_frames = []
    importance_rows = []
    selected_model = None

    for feature_set, features in FEATURE_SETS.items():

        for model_name in MODEL_NAMES:

            print(f"\nTesting 2025: {feature_set} | {model_name}")

            model = create_model(model_name)

            model.fit(
                train_validation[features],
                train_validation["LST_C"],
            )

            predictions = model.predict(
                test[features]
            )

            is_selected = (
                feature_set == best_feature_set
                and model_name == best_model_name
            )

            test_results.append({
                "feature_set": feature_set,
                "model": model_name,
                "feature_count": len(features),
                "selected_using_2024_validation": is_selected,
                **calculate_metrics(
                    test["LST_C"], predictions
                ),
            })

            # Save predictions for scene-level error analysis.
            metadata = [
                column for column in [
                    "date",
                    "year",
                    "product_id",
                    "longitude",
                    "latitude",
                    "utm_easting_m",
                    "utm_northing_m",
                ]
                if column in test.columns
            ]

            frame = test[metadata].copy()

            frame["feature_set"] = feature_set
            frame["model"] = model_name
            frame["LST_C_observed"] = test["LST_C"].to_numpy()
            frame["LST_C_predicted"] = predictions

            frame["residual_C"] = (
                frame["LST_C_observed"]
                - frame["LST_C_predicted"]
            )

            prediction_frames.append(frame)

            if model_name == "Random Forest":
                for feature, importance in zip(
                    features, model.feature_importances_
                ):
                    importance_rows.append({
                        "feature_set": feature_set,
                        "feature": feature,
                        "importance": importance,
                    })

            if is_selected:
                selected_model = model

    test_df = pd.DataFrame(test_results)

    print("\nHELD-OUT TEST RESULTS — 2025")
    print("-" * 65)

    print(test_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))


    # ==================================================
    # 6. SAVE OUTPUTS
    # ==================================================

    validation_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Validation_Metrics.csv"
    )

    test_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Test_Metrics.csv"
    )

    predictions_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Test_Predictions.csv"
    )

    importance_file = (
        FOLDER / "Bengaluru_UHI_Temporal_RF_Importance.csv"
    )

    model_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Selected_Model.joblib"
    )

    audit_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Run_Audit.json"
    )

    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)

    pd.concat(
        prediction_frames, ignore_index=True
    ).to_csv(predictions_file, index=False)

    pd.DataFrame(importance_rows).to_csv(
        importance_file, index=False
    )

    joblib.dump(selected_model, model_file)

    audit = {
        "input_file": INPUT_FILE.name,
        "temporal_split_file": split_file.name,
        "training_years": [2021, 2022, 2023],
        "validation_year": 2024,
        "test_year": 2025,
        "selected_feature_set": best_feature_set,
        "selected_model": best_model_name,
        "selection_criterion": "2024 validation RMSE",
        "test_scenes": int(test["product_id"].nunique()),
        "test_metrics": test_df[
            (
                (test_df["feature_set"] == best_feature_set)
                & (test_df["model"] == best_model_name)
            )
        ].iloc[0].to_dict(),
    }

    with open(audit_file, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4)

    print("\nFILES CREATED")
    print("-" * 65)

    for path in [
        split_file,
        validation_file,
        test_file,
        predictions_file,
        importance_file,
        model_file,
        audit_file,
    ]:
        print(path.name)

    print("\nTemporal holdout experiment completed.")


if __name__ == "__main__":
    main()
