
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

DATA_FILE = (
    FOLDER / "Bengaluru_UHI_Scene_Split_By_Year_2021_2025.csv"
)

PREDICTIONS_FILE = (
    FOLDER / "Bengaluru_UHI_Scene_Test_Predictions.csv"
)

OUTPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Scene_Shift_Diagnostic.csv"
)


# ==================================================
# 2. LOAD FILES
# ==================================================

def main():

    for file in [DATA_FILE, PREDICTIONS_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file.name}"
            )

    data = pd.read_csv(DATA_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )

    required_data = [
        "date",
        "product_id",
        "scene_split",
        "LST_C",
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
    ]

    required_predictions = [
        "product_id",
        "date",
        "LST_C_observed",
        "LST_C_predicted",
    ]

    for columns, frame, label in [
        (required_data, data, "scene dataset"),
        (required_predictions, predictions, "predictions"),
    ]:
        missing = [
            column for column in columns
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {label}: {missing}"
            )

    data["scene_split"] = (
        data["scene_split"]
        .astype(str).str.strip().str.lower()
    )

    numeric_data = [
        "LST_C",
        "NDVI",
        "NDMI",
        "builtup_fraction_500m_2021",
        "elevation_m",
    ]

    for column in numeric_data:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    for column in [
        "LST_C_observed",
        "LST_C_predicted",
    ]:
        predictions[column] = pd.to_numeric(
            predictions[column], errors="coerce"
        )

    data = data.dropna(
        subset=required_data
    ).copy()

    predictions = predictions.dropna(
        subset=required_predictions
    ).copy()

    predictions["residual_C"] = (
        predictions["LST_C_observed"]
        - predictions["LST_C_predicted"]
    )

    predictions["absolute_error_C"] = (
        predictions["residual_C"].abs()
    )

    predictions["squared_error_C"] = (
        predictions["residual_C"] ** 2
    )

    print("SCENE SHIFT DIAGNOSTIC")
    print("-" * 60)
    print(f"Dataset rows: {len(data):,}")
    print(f"Prediction rows: {len(predictions):,}")


    # ==================================================
    # 3. SUMMARIZE EACH SCENE
    # ==================================================

    scene_summary = (
        data.groupby(
            ["product_id", "date", "scene_split"],
            as_index=False,
        )
        .agg(
            scene_samples=("LST_C", "size"),
            LST_mean_C=("LST_C", "mean"),
            LST_std_C=("LST_C", "std"),
            NDVI_mean=("NDVI", "mean"),
            NDMI_mean=("NDMI", "mean"),
            builtup_mean=(
                "builtup_fraction_500m_2021", "mean"
            ),
            elevation_mean_m=("elevation_m", "mean"),
        )
    )


    # ==================================================
    # 4. CALCULATE TRAINING-SCENE BASELINES
    # ==================================================

    comparison_columns = [
        "LST_mean_C",
        "NDVI_mean",
        "NDMI_mean",
        "builtup_mean",
        "elevation_mean_m",
    ]

    training_scenes = scene_summary[
        scene_summary["scene_split"] == "train"
    ]

    baseline_mean = training_scenes[
        comparison_columns
    ].mean()

    baseline_std = training_scenes[
        comparison_columns
    ].std(ddof=1)

    print("\nTRAINING-SCENE BASELINES")
    print("-" * 60)

    for column in comparison_columns:
        print(
            f"{column}: mean scene value = "
            f"{baseline_mean[column]:.4f}, "
            f"between-scene SD = {baseline_std[column]:.4f}"
        )

    # Standardized distance from the training-scene
    # distribution. These are diagnostic indicators,
    # not formal statistical significance tests.

    for column in comparison_columns:

        sd = baseline_std[column]

        if pd.isna(sd) or sd == 0:
            scene_summary[f"z_{column}"] = np.nan
        else:
            scene_summary[f"z_{column}"] = (
                scene_summary[column] - baseline_mean[column]
            ) / sd


    # ==================================================
    # 5. SUMMARIZE TEST PREDICTION ERRORS BY SCENE
    # ==================================================

    test_errors = (
        predictions.groupby("product_id", as_index=False)
        .agg(
            prediction_date=("date", "first"),
            test_rows=("residual_C", "size"),
            observed_mean_C=("LST_C_observed", "mean"),
            predicted_mean_C=("LST_C_predicted", "mean"),
            mean_error_C=("residual_C", "mean"),
            MAE_C=("absolute_error_C", "mean"),
            mean_squared_error_C=("squared_error_C", "mean"),
        )
    )

    test_errors["RMSE_C"] = np.sqrt(
        test_errors["mean_squared_error_C"]
    )

    test_errors = test_errors.drop(
        columns=["mean_squared_error_C"]
    )

    # Confirm predictions match the test scenes in the input.
    expected_test_scenes = set(
        scene_summary.loc[
            scene_summary["scene_split"] == "test",
            "product_id",
        ]
    )

    predicted_test_scenes = set(
        test_errors["product_id"]
    )

    if expected_test_scenes != predicted_test_scenes:
        raise ValueError(
            "Prediction product IDs do not match the "
            "test scenes in the scene-split dataset."
        )

    result = scene_summary.merge(
        test_errors,
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    result["date"] = result["date"].dt.strftime(
        "%Y-%m-%d"
    )

    result = result.sort_values(
        ["scene_split", "date"]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # ==================================================
    # 6. PRINT TEST SCENES BY ERROR
    # ==================================================

    test_result = result[
        result["scene_split"] == "test"
    ].copy()

    test_result = test_result.sort_values(
        "MAE_C", ascending=False
    )

    display_columns = [
        "date",
        "scene_samples",
        "observed_mean_C",
        "predicted_mean_C",
        "mean_error_C",
        "MAE_C",
        "RMSE_C",
        "z_LST_mean_C",
        "z_NDVI_mean",
        "z_NDMI_mean",
    ]

    print("\nTEST SCENES: LARGEST ERRORS FIRST")
    print("-" * 110)

    print(
        test_result[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nSaved diagnostic:")
    print(OUTPUT_FILE.name)

    print("\nScene shift diagnostic completed.")


if __name__ == "__main__":
    main()

