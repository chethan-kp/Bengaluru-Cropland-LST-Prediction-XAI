
from pathlib import Path
import hashlib
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

INPUT_FILE = FOLDER / "Bengaluru_UHI_Spatial_Split_2021_2025.csv"

TARGET = "LST_C"

FEATURES = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos",
]

EXPECTED_SPLITS = {"train", "validation", "test"}


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def calculate_metrics(actual, predicted):
    return {
        "RMSE_C": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "MAE_C": float(
            mean_absolute_error(actual, predicted)
        ),
        "R2": float(r2_score(actual, predicted)),
    }


def create_model(model_name):
    if model_name == "Linear Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            max_features=1.0,
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ==================================================
# 3. LOAD AND VALIDATE DATA
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run create_spatial_split.py first."
        )

    # Print the exact file used
    print("INPUT FILE")
    print("-" * 60)
    print(INPUT_FILE.resolve())

    # Calculate a fingerprint to identify this exact CSV file
    file_hash = hashlib.sha256(
        INPUT_FILE.read_bytes()
    ).hexdigest()

    print(f"SHA-256: {file_hash}")

    data = pd.read_csv(INPUT_FILE)

    required_columns = FEATURES + [TARGET, "split"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The input dataset is empty.")

    if data["split"].isna().any():
        raise ValueError("Missing split labels were found.")

    data["split"] = (
        data["split"].astype(str).str.strip().str.lower()
    )

    actual_splits = set(data["split"].unique())

    if actual_splits != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected splits {EXPECTED_SPLITS}, "
            f"but found {actual_splits}"
        )

    # Ensure model columns are numeric
    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    model_values = data[FEATURES + [TARGET]]

    if model_values.isna().any().any():
        bad_columns = model_values.columns[
            model_values.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing or non-numeric values in: {bad_columns}"
        )

    if not np.isfinite(model_values.to_numpy(dtype=float)).all():
        raise ValueError(
            "Infinite values found in the model columns."
        )

    # Confirm that spatial blocks do not cross partitions
    if {"block_x", "block_y"}.issubset(data.columns):

        block_split_counts = (
            data.groupby(["block_x", "block_y"])["split"]
            .nunique()
        )

        crossing_blocks = int(
            (block_split_counts > 1).sum()
        )

        print(f"Spatial blocks crossing splits: {crossing_blocks}")

        if crossing_blocks != 0:
            raise ValueError(
                "Spatial blocks cross splits. "
                "Rebuild the spatial split before training."
            )

    # ==================================================
    # 4. CREATE TRAIN / VALIDATION / TEST SETS
    # ==================================================

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Training, validation, or test data is empty."
        )

    print("\nDATASET SUMMARY")
    print("-" * 60)
    print(f"Total rows:      {len(data):,}")
    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")
    print(f"Features: {FEATURES}")

    print("\nLST SUMMARY BY PARTITION")
    print("-" * 60)

    print(
        data.groupby("split")[TARGET]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    # ==================================================
    # 5. TRAIN AND COMPARE BASELINE MODELS
    # ==================================================

    validation_results = []
    candidate_names = [
        "Linear Regression",
        "Random Forest",
    ]

    fitted_candidates = {}

    for model_name in candidate_names:

        print(f"\nTraining {model_name}...")

        model = create_model(model_name)
        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)

        metrics = calculate_metrics(
            y_val, val_predictions
        )

        validation_results.append({
            "model": model_name,
            **metrics,
        })

        fitted_candidates[model_name] = model

    validation_df = pd.DataFrame(
        validation_results
    ).sort_values("RMSE_C").reset_index(drop=True)

    print("\nVALIDATION RESULTS")
    print("-" * 60)
    print(validation_df.to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))

    # Select using validation RMSE only
    best_name = validation_df.iloc[0]["model"]

    print(f"\nSelected model: {best_name}")
    print("Selection criterion: lowest validation RMSE")

    # ==================================================
    # 6. RETRAIN SELECTED MODEL ON TRAIN + VALIDATION
    # ==================================================

    # The test set remains completely untouched until evaluation.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    X_train_validation = train_validation[FEATURES]
    y_train_validation = train_validation[TARGET]

    final_model = create_model(best_name)

    print(
        f"\nRetraining {best_name} on "
        f"{len(train_validation):,} training + validation rows..."
    )

    final_model.fit(
        X_train_validation,
        y_train_validation,
    )

    # ==================================================
    # 7. FINAL HELD-OUT TEST EVALUATION
    # ==================================================

    X_test = test[FEATURES]
    y_test = test[TARGET]

    test_predictions = final_model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 60)
    print(f"Selected model: {best_name}")
    print(f"Test rows:      {len(test):,}")
    print(f"RMSE_C:         {test_metrics['RMSE_C']:.4f}")
    print(f"MAE_C:          {test_metrics['MAE_C']:.4f}")
    print(f"R2:             {test_metrics['R2']:.4f}")

    # ==================================================
    # 8. SAVE VALIDATION AND TEST METRICS
    # ==================================================

    validation_path = (
        FOLDER / "Bengaluru_UHI_Validation_Metrics.csv"
    )

    test_metrics_path = (
        FOLDER / "Bengaluru_UHI_Test_Metrics.csv"
    )

    validation_df.to_csv(
        validation_path, index=False
    )

    pd.DataFrame([{
        "model": best_name,
        "training_rows_used_for_final_model": len(
            train_validation
        ),
        "test_rows": len(test),
        **test_metrics,
    }]).to_csv(
        test_metrics_path, index=False
    )

    # ==================================================
    # 9. SAVE TEST PREDICTIONS
    # ==================================================

    predictions_df = test.copy()

    predictions_df = predictions_df.rename(
        columns={TARGET: "LST_C_observed"}
    )

    predictions_df["LST_C_predicted"] = test_predictions

    predictions_df["residual_C"] = (
        predictions_df["LST_C_observed"]
        - predictions_df["LST_C_predicted"]
    )

    predictions_df["absolute_error_C"] = (
        predictions_df["residual_C"].abs()
    )

    predictions_path = (
        FOLDER / "Bengaluru_UHI_Test_Predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path, index=False
    )

    # ==================================================
    # 10. SAVE FEATURE IMPORTANCE IF RANDOM FOREST WINS
    # ==================================================

    importance_path = (
        FOLDER / "Bengaluru_UHI_Random_Forest_Feature_Importance.csv"
    )

    if best_name == "Random Forest":

        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": final_model.feature_importances_,
        }).sort_values(
            "importance", ascending=False
        )

        importance_df.to_csv(
            importance_path, index=False
        )

        print("\nRANDOM FOREST FEATURE IMPORTANCE")
        print("-" * 60)
        print(importance_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        ))

    elif importance_path.exists():
        importance_path.unlink()

    # ==================================================
    # 11. SAVE FINAL MODEL AND RUN AUDIT
    # ==================================================

    model_path = (
        FOLDER / "Bengaluru_UHI_Best_Baseline_Model.joblib"
    )

    joblib.dump(final_model, model_path)

    audit = {
        "input_file": str(INPUT_FILE.resolve()),
        "input_sha256": file_hash,
        "total_rows": int(len(data)),
        "training_rows_original": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "final_model_training_rows": int(len(train_validation)),
        "features": FEATURES,
        "target": TARGET,
        "selected_model": best_name,
        "selection_metric": "validation RMSE",
        "test_metrics": test_metrics,
    }

    audit_path = (
        FOLDER / "Bengaluru_UHI_Model_Run_Audit.json"
    )

    with open(audit_path, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4)

    print("\nFILES CREATED")
    print("-" * 60)

    for path in [
        validation_path,
        test_metrics_path,
        predictions_path,
        model_path,
        audit_path,
    ]:
        print(path.name)

    if best_name == "Random Forest":
        print(importance_path.name)

    print("\nBaseline modelling completed successfully.")


if __name__ == "__main__":
    main()
