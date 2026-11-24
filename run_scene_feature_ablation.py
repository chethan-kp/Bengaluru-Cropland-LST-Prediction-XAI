
from pathlib import Path

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


FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Scene_Split_By_Year_2021_2025.csv"
)

FEATURE_SETS = {
    "Full model": [
        "NDVI", "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m", "year",
        "doy_sin", "doy_cos",
    ],
    "Without day-of-year": [
        "NDVI", "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m", "year",
    ],
    "Environmental only": [
        "NDVI", "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
    ],
}


def make_model(name):
    if name == "Linear Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    return RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=5,
        max_features=1.0,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )


def metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "samples": len(actual),
        "RMSE_C": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAE_C": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "mean_error_C": np.mean(actual - predicted),
    }


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "date", "product_id", "scene_split", "LST_C",
        "NDVI", "NDMI", "builtup_fraction_500m_2021",
        "elevation_m", "year",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid acquisition dates found.")

    # Match the seasonal-feature convention used in
    # train_scene_heldout_models.py.
    day = data["date"].dt.dayofyear
    angle = 2 * np.pi * day / 365.25
    data["doy_sin"] = np.sin(angle)
    data["doy_cos"] = np.cos(angle)

    data["scene_split"] = (
        data["scene_split"].astype(str).str.strip().str.lower()
    )

    all_features = sorted({
        feature
        for feature_list in FEATURE_SETS.values()
        for feature in feature_list
    })

    for column in all_features + ["LST_C"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    numeric = data[all_features + ["LST_C"]]

    if numeric.isna().any().any():
        raise ValueError("Missing or invalid model values found.")

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Infinite model values found.")

    # Verify scene separation.
    for column in ["product_id", "date"]:

        crossing = data.groupby(column)["scene_split"].nunique()

        if (crossing > 1).any():
            raise ValueError(
                f"{column} crosses data partitions."
            )

    train = data[data["scene_split"] == "train"].copy()
    validation = data[
        data["scene_split"] == "validation"
    ].copy()
    test = data[data["scene_split"] == "test"].copy()

    train_validation = pd.concat(
        [train, validation], ignore_index=True
    )

    if train.empty or validation.empty or test.empty:
        raise ValueError("A data partition is empty.")

    print("SCENE-HELD-OUT FEATURE ABLATION")
    print("-" * 65)
    print(
        f"Training: {len(train):,} rows, "
        f"{train['product_id'].nunique()} scenes"
    )
    print(
        f"Validation: {len(validation):,} rows, "
        f"{validation['product_id'].nunique()} scenes"
    )
    print(
        f"Test: {len(test):,} rows, "
        f"{test['product_id'].nunique()} scenes"
    )

    validation_results = []
    test_results = []
    prediction_results = []

    # Compare both model types for every feature set.
    for feature_name, features in FEATURE_SETS.items():

        print(f"\nFeature set: {feature_name}")

        candidates = []

        for model_name in ["Linear Regression", "Random Forest"]:

            print(f"  Training {model_name} on training scenes...")

            model = make_model(model_name)

            model.fit(train[features], train["LST_C"])

            val_prediction = model.predict(validation[features])

            result = {
                "feature_set": feature_name,
                "model": model_name,
                **metrics(validation["LST_C"], val_prediction),
            }

            validation_results.append(result)
            candidates.append(result)

        # Select model type using validation RMSE only.
        best = min(candidates, key=lambda item: item["RMSE_C"])
        selected_name = best["model"]

        print(
            f"  Selected on validation: {selected_name} "
            f"(RMSE {best['RMSE_C']:.4f} °C)"
        )

        # Fit the chosen model on train + validation.
        final_model = make_model(selected_name)

        final_model.fit(
            train_validation[features],
            train_validation["LST_C"],
        )

        test_prediction = final_model.predict(test[features])

        test_result = {
            "feature_set": feature_name,
            "selected_model": selected_name,
            "feature_count": len(features),
            **metrics(test["LST_C"], test_prediction),
        }

        test_results.append(test_result)

        # Store predictions for later analysis.
        metadata = [
            c for c in [
                "date", "year", "product_id",
                "longitude", "latitude",
            ]
            if c in test.columns
        ]

        frame = test[metadata].copy()
        frame["feature_set"] = feature_name
        frame["selected_model"] = selected_name
        frame["LST_C_observed"] = test["LST_C"].to_numpy()
        frame["LST_C_predicted"] = test_prediction
        frame["residual_C"] = (
            test["LST_C"].to_numpy() - test_prediction
        )

        prediction_results.append(frame)

    # Print and save validation results.
    validation_df = pd.DataFrame(validation_results).sort_values(
        ["feature_set", "RMSE_C"]
    )

    test_df = pd.DataFrame(test_results).sort_values("RMSE_C")

    print("\nVALIDATION RESULTS")
    print("-" * 65)
    print(validation_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print("\nSCENE-HELD-OUT TEST RESULTS")
    print("-" * 65)
    print(test_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    validation_df.to_csv(
        FOLDER / "Bengaluru_UHI_Scene_Ablation_Validation.csv",
        index=False,
    )

    test_df.to_csv(
        FOLDER / "Bengaluru_UHI_Scene_Ablation_Test.csv",
        index=False,
    )

    pd.concat(
        prediction_results, ignore_index=True
    ).to_csv(
        FOLDER / "Bengaluru_UHI_Scene_Ablation_Predictions.csv",
        index=False,
    )

    print("\nCreated:")
    print("Bengaluru_UHI_Scene_Ablation_Validation.csv")
    print("Bengaluru_UHI_Scene_Ablation_Test.csv")
    print("Bengaluru_UHI_Scene_Ablation_Predictions.csv")

    print("\nScene-held-out feature ablation completed.")


if __name__ == "__main__":
    main()

