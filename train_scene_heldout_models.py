
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

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

FEATURES = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos",
]

TARGET = "LST_C"


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def calculate_metrics(actual, predicted):
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
            np.mean(np.asarray(actual) - np.asarray(predicted))
        ),
    }


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


# ==================================================
# 3. LOAD DATA AND CREATE MISSING FEATURES
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    required_base = [
        "date",
        "product_id",
        "scene_split",
        "LST_C",
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
        "year",
    ]

    missing = [
        col for col in required_base
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Derive seasonal features directly from acquisition dates.
    dates = pd.to_datetime(data["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError(
            "Invalid dates found in the scene-split dataset."
        )

    day_of_year = dates.dt.dayofyear

    # Use a consistent annual cycle for the sine/cosine features.
    angle = 2 * np.pi * day_of_year / 365.25

    data["doy_sin"] = np.sin(angle)
    data["doy_cos"] = np.cos(angle)

    # Normalize split labels.
    data["scene_split"] = (
        data["scene_split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if set(data["scene_split"].unique()) != {
        "train", "validation", "test"
    }:
        raise ValueError(
            "Expected train, validation, and test partitions."
        )

    # Check that complete scenes and dates stay in one partition.
    for column in ["product_id", "date"]:

        crossing = (
            data.groupby(column)["scene_split"].nunique()
        )

        violations = int((crossing > 1).sum())

        print(f"{column} values crossing splits: {violations}")

        if violations:
            raise ValueError(
                f"Leakage detected: {column} crosses partitions."
            )

    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    numeric = data[FEATURES + [TARGET]]

    if numeric.isna().any().any():
        raise ValueError(
            "Missing or invalid model values found."
        )

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "Infinite values found in model features."
        )

    train = data[data["scene_split"] == "train"].copy()
    validation = data[
        data["scene_split"] == "validation"
    ].copy()
    test = data[data["scene_split"] == "test"].copy()

    print("\nSCENE-HELD-OUT DATASET")
    print("-" * 60)

    for name, subset in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            f"{name}: {len(subset):,} rows, "
            f"{subset['product_id'].nunique()} scenes"
        )

    # ==================================================
    # 4. COMPARE MODELS ON VALIDATION SCENES
    # ==================================================

    results = []

    for name in ["Linear Regression", "Random Forest"]:

        print(f"\nTraining {name}...")

        model = create_model(name)

        model.fit(train[FEATURES], train[TARGET])

        predictions = model.predict(validation[FEATURES])

        results.append({
            "model": name,
            **calculate_metrics(
                validation[TARGET], predictions
            ),
        })

    validation_df = pd.DataFrame(results).sort_values(
        "RMSE_C"
    ).reset_index(drop=True)

    print("\nVALIDATION RESULTS")
    print("-" * 60)
    print(validation_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    best_name = validation_df.iloc[0]["model"]

    print(f"\nSelected model: {best_name}")
    print("Selection criterion: lowest validation RMSE")

    # ==================================================
    # 5. REFIT ON TRAINING + VALIDATION SCENES
    # ==================================================

    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    final_model = create_model(best_name)

    final_model.fit(
        train_validation[FEATURES],
        train_validation[TARGET],
    )

    # ==================================================
    # 6. EVALUATE ON UNSEEN TEST SCENES
    # ==================================================

    test_predictions = final_model.predict(
        test[FEATURES]
    )

    test_metrics = calculate_metrics(
        test[TARGET], test_predictions
    )

    print("\nSCENE-HELD-OUT TEST RESULTS")
    print("-" * 60)
    print(f"Model: {best_name}")
    print(f"Test rows: {len(test):,}")
    print(f"Test scenes: {test['product_id'].nunique()}")

    for metric, value in test_metrics.items():
        if metric == "samples":
            print(f"{metric}: {value:,}")
        else:
            print(f"{metric}: {value:.4f}")

    # ==================================================
    # 7. SAVE METRICS
    # ==================================================

    validation_file = (
        FOLDER / "Bengaluru_UHI_Scene_Validation_Metrics.csv"
    )

    test_file = (
        FOLDER / "Bengaluru_UHI_Scene_Test_Metrics.csv"
    )

    validation_df.to_csv(validation_file, index=False)

    pd.DataFrame([{
        "model": best_name,
        "training_plus_validation_rows": len(
            train_validation
        ),
        "test_rows": len(test),
        "test_scenes": int(
            test["product_id"].nunique()
        ),
        **test_metrics,
    }]).to_csv(test_file, index=False)

    # ==================================================
    # 8. SAVE TEST PREDICTIONS
    # ==================================================

    metadata = [
        col for col in [
            "date", "year", "product_id",
            "longitude", "latitude",
            "utm_easting_m", "utm_northing_m",
            "block_x", "block_y",
        ]
        if col in test.columns
    ]

    predictions_df = test[metadata].copy()

    predictions_df["LST_C_observed"] = (
        test[TARGET].to_numpy()
    )

    predictions_df["LST_C_predicted"] = test_predictions

    predictions_df["residual_C"] = (
        test[TARGET].to_numpy() - test_predictions
    )

    predictions_df["absolute_error_C"] = (
        predictions_df["residual_C"].abs()
    )

    prediction_file = (
        FOLDER / "Bengaluru_UHI_Scene_Test_Predictions.csv"
    )

    predictions_df.to_csv(prediction_file, index=False)

    # ==================================================
    # 9. SAVE MODEL, IMPORTANCE, AND AUDIT
    # ==================================================

    model_file = (
        FOLDER / "Bengaluru_UHI_Scene_Heldout_Model.joblib"
    )

    joblib.dump(final_model, model_file)

    if best_name == "Random Forest":

        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance_df.to_csv(
            FOLDER / "Bengaluru_UHI_Scene_Feature_Importance.csv",
            index=False,
        )

        print("\nFEATURE IMPORTANCE")
        print("-" * 60)
        print(importance_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ))

    audit = {
        "input_file": INPUT_FILE.name,
        "features": FEATURES,
        "target": TARGET,
        "seasonal_features_created_from_date": True,
        "seasonal_cycle_days": 365.25,
        "selected_model": best_name,
        "selection_metric": "validation RMSE",
        "training_scenes": int(
            train["product_id"].nunique()
        ),
        "validation_scenes": int(
            validation["product_id"].nunique()
        ),
        "test_scenes": int(
            test["product_id"].nunique()
        ),
        "test_metrics": test_metrics,
    }

    audit_file = (
        FOLDER / "Bengaluru_UHI_Scene_Model_Audit.json"
    )

    with open(audit_file, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4)

    print("\nScene-held-out modelling completed.")

    print("\nOutput files:")
    print(validation_file.name)
    print(test_file.name)
    print(prediction_file.name)
    print(model_file.name)
    print(audit_file.name)


if __name__ == "__main__":
    main()
