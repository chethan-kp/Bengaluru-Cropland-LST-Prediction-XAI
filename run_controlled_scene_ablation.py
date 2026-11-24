
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
# 3. LOAD AND CHECK THE DATA
# ==================================================

def main():

    print("CONTROLLED SCENE-HELD-OUT ABLATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE.name}"
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "date", "product_id", "scene_split", "LST_C",
        "NDVI", "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m", "year",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    if data["date"].isna().any():
        raise ValueError("Invalid acquisition dates found.")

    # Recreate the same seasonal features used by the
    # previous scene-held-out training script.
    day = data["date"].dt.dayofyear
    angle = 2 * np.pi * day / 365.25

    data["doy_sin"] = np.sin(angle)
    data["doy_cos"] = np.cos(angle)

    data["scene_split"] = (
        data["scene_split"].astype(str).str.strip().str.lower()
    )

    all_features = sorted({
        f for feature_list in FEATURE_SETS.values()
        for f in feature_list
    })

    for column in all_features + ["LST_C"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    values = data[all_features + ["LST_C"]]

    if values.isna().any().any():
        raise ValueError("Missing or invalid model values found.")

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Infinite model values found.")

    if set(data["scene_split"].unique()) != {
        "train", "validation", "test"
    }:
        raise ValueError("Unexpected scene split labels.")

    # Check that dates and scenes never cross partitions.
    for column in ["date", "product_id"]:
        crossings = (
            data.groupby(column)["scene_split"].nunique()
        )

        count = int((crossings > 1).sum())

        print(f"{column} values crossing splits: {count}")

        if count:
            raise ValueError(
                f"Data leakage detected for {column}."
            )

    train = data[data["scene_split"] == "train"].copy()
    validation = data[
        data["scene_split"] == "validation"
    ].copy()
    test = data[data["scene_split"] == "test"].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("A data partition is empty.")

    train_validation = pd.concat(
        [train, validation], ignore_index=True
    )

    print(f"\nTraining: {len(train):,} rows, "
          f"{train['product_id'].nunique()} scenes")
    print(f"Validation: {len(validation):,} rows, "
          f"{validation['product_id'].nunique()} scenes")
    print(f"Test: {len(test):,} rows, "
          f"{test['product_id'].nunique()} scenes")


    # ==================================================
    # 4. VALIDATION: COMPARE ALL SIX COMBINATIONS
    # ==================================================

    validation_results = []

    for feature_set, features in FEATURE_SETS.items():

        for model_name in MODEL_NAMES:

            print(f"\nValidation: {feature_set} | {model_name}")

            model = create_model(model_name)

            model.fit(train[features], train["LST_C"])

            predictions = model.predict(validation[features])

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

    print("\nVALIDATION RESULTS")
    print("-" * 65)
    print(validation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    # Select using validation RMSE, not test results.
    best = validation_df.iloc[0]

    best_feature_set = best["feature_set"]
    best_model_name = best["model"]

    print("\nSELECTED CONFIGURATION")
    print("-" * 65)
    print(f"Features: {best_feature_set}")
    print(f"Model: {best_model_name}")
    print(f"Validation RMSE: {best['RMSE_C']:.4f} °C")


    # ==================================================
    # 5. FIT ON TRAIN + VALIDATION, THEN TEST ALL SIX
    # ==================================================

    test_results = []
    predictions_saved = []
    importance_saved = []
    selected_model = None

    for feature_set, features in FEATURE_SETS.items():

        for model_name in MODEL_NAMES:

            print(f"\nTest evaluation: {feature_set} | {model_name}")

            model = create_model(model_name)

            # Test scenes are never used for fitting.
            model.fit(
                train_validation[features],
                train_validation["LST_C"],
            )

            predictions = model.predict(test[features])

            test_results.append({
                "feature_set": feature_set,
                "model": model_name,
                "feature_count": len(features),
                "selected_by_validation": (
                    feature_set == best_feature_set
                    and model_name == best_model_name
                ),
                **calculate_metrics(
                    test["LST_C"], predictions
                ),
            })

            metadata = [
                c for c in [
                    "date", "year", "product_id",
                    "longitude", "latitude",
                ]
                if c in test.columns
            ]

            saved = test[metadata].copy()
            saved["feature_set"] = feature_set
            saved["model"] = model_name
            saved["LST_C_observed"] = test["LST_C"].to_numpy()
            saved["LST_C_predicted"] = predictions
            saved["residual_C"] = (
                test["LST_C"].to_numpy() - predictions
            )

            predictions_saved.append(saved)

            if model_name == "Random Forest":
                for feature, importance in zip(
                    features, model.feature_importances_
                ):
                    importance_saved.append({
                        "feature_set": feature_set,
                        "feature": feature,
                        "importance": importance,
                    })

            if (
                feature_set == best_feature_set
                and model_name == best_model_name
            ):
                selected_model = model

    test_df = pd.DataFrame(test_results)

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 65)
    print(test_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))


    # ==================================================
    # 6. SAVE RESULTS
    # ==================================================

    validation_file = (
        FOLDER / "Bengaluru_UHI_Controlled_Ablation_Validation.csv"
    )
    test_file = (
        FOLDER / "Bengaluru_UHI_Controlled_Ablation_Test.csv"
    )
    predictions_file = (
        FOLDER / "Bengaluru_UHI_Controlled_Ablation_Predictions.csv"
    )
    importance_file = (
        FOLDER / "Bengaluru_UHI_Controlled_Ablation_RF_Importance.csv"
    )
    model_file = (
        FOLDER / "Bengaluru_UHI_Controlled_Ablation_Selected_Model.joblib"
    )
    audit_file = (
        FOLDER / "Bengaluru_UHI_Controlled_Ablation_Audit.json"
    )

    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)

    pd.concat(
        predictions_saved, ignore_index=True
    ).to_csv(predictions_file, index=False)

    pd.DataFrame(importance_saved).to_csv(
        importance_file, index=False
    )

    joblib.dump(selected_model, model_file)

    audit = {
        "split_design": "scene-held-out",
        "input_file": INPUT_FILE.name,
        "seasonal_formula": "sin/cos(2*pi*day_of_year/365.25)",
        "selected_feature_set": best_feature_set,
        "selected_model": best_model_name,
        "selection_metric": "validation RMSE",
        "training_scenes": int(train["product_id"].nunique()),
        "validation_scenes": int(
            validation["product_id"].nunique()
        ),
        "test_scenes": int(test["product_id"].nunique()),
    }

    with open(audit_file, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4)

    print("\nFILES CREATED")
    print("-" * 65)

    for path in [
        validation_file,
        test_file,
        predictions_file,
        importance_file,
        model_file,
        audit_file,
    ]:
        print(path.name)

    print("\nControlled scene-held-out ablation completed successfully.")


if __name__ == "__main__":
    main()
