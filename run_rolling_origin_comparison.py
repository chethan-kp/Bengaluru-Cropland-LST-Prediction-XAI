
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
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

RESULTS_FILE = (
    FOLDER / "Bengaluru_UHI_Rolling_Origin_Results.csv"
)

PREDICTIONS_FILE = (
    FOLDER / "Bengaluru_UHI_Rolling_Origin_Predictions.csv"
)

TEST_YEARS = [2023, 2024, 2025]

FEATURES = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos",
]


# ==================================================
# 2. METRIC FUNCTION
# ==================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "samples": len(actual),
        "RMSE_C": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAE_C": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "mean_error_C": np.mean(actual - predicted),
    }


# ==================================================
# 3. LOAD AND PREPARE DATA
# ==================================================

def main():

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

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    if data["date"].isna().any():
        raise ValueError("Invalid acquisition dates found.")

    # Derive temporal features consistently.
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    day = data["date"].dt.dayofyear
    angle = 2 * np.pi * day / 365.25

    data["doy_sin"] = np.sin(angle)
    data["doy_cos"] = np.cos(angle)

    for column in FEATURES + ["LST_C"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    numeric = data[FEATURES + ["LST_C"]]

    if numeric.isna().any().any():
        raise ValueError("Missing or invalid numeric values found.")

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Infinite values found.")

    if set(TEST_YEARS) - set(data["year"].unique()):
        raise ValueError("One or more requested test years are missing.")

    print("ROLLING-ORIGIN TEMPORAL EVALUATION")
    print("-" * 65)
    print(f"Input rows: {len(data):,}")
    print(f"Test years: {TEST_YEARS}")


    # ==================================================
    # 4. TRAIN ON PAST YEARS, TEST ON THE NEXT YEAR
    # ==================================================

    all_metrics = []
    all_predictions = []

    for test_year in TEST_YEARS:

        history = data[data["year"] < test_year].copy()
        test = data[data["year"] == test_year].copy()

        if history.empty or test.empty:
            print(f"Skipping {test_year}: missing history or test data.")
            continue

        print(f"\nTest year: {test_year}")
        print(f"Historical rows: {len(history):,}")
        print(f"Historical scenes: {history['product_id'].nunique()}")
        print(f"Test rows: {len(test):,}")
        print(f"Test scenes: {test['product_id'].nunique()}")

        # --------------------------------------------------
        # A. Fit Linear Regression using only past years
        # --------------------------------------------------

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        model.fit(
            history[FEATURES],
            history["LST_C"],
        )

        model_predictions = model.predict(
            test[FEATURES]
        )

        # --------------------------------------------------
        # B. Monthly climatology using only past years
        # --------------------------------------------------

        historical_monthly_means = (
            history.groupby("month")["LST_C"].mean()
        )

        historical_overall_mean = history["LST_C"].mean()

        climatology_predictions = (
            test["month"]
            .map(historical_monthly_means)
            .fillna(historical_overall_mean)
            .to_numpy()
        )

        # --------------------------------------------------
        # C. Metrics for both approaches
        # --------------------------------------------------

        year_results = [
            (
                "Linear Regression",
                model_predictions,
            ),
            (
                "Monthly climatology",
                climatology_predictions,
            ),
        ]

        for approach, predictions in year_results:

            metrics = calculate_metrics(
                test["LST_C"],
                predictions,
            )

            all_metrics.append({
                "test_year": test_year,
                "approach": approach,
                "historical_years": (
                    f"{int(history['year'].min())}-"
                    f"{int(history['year'].max())}"
                ),
                "historical_scenes": int(
                    history["product_id"].nunique()
                ),
                "test_scenes": int(
                    test["product_id"].nunique()
                ),
                **metrics,
            })

        # --------------------------------------------------
        # D. Save individual predictions
        # --------------------------------------------------

        metadata = [
            c for c in [
                "date",
                "year",
                "month",
                "product_id",
                "longitude",
                "latitude",
            ]
            if c in test.columns
        ]

        saved = test[metadata].copy()

        saved["test_year"] = test_year
        saved["LST_C_observed"] = test["LST_C"].to_numpy()

        saved["Linear_Regression_predicted_C"] = (
            model_predictions
        )

        saved["Monthly_climatology_predicted_C"] = (
            climatology_predictions
        )

        saved["Linear_Regression_error_C"] = (
            saved["LST_C_observed"]
            - saved["Linear_Regression_predicted_C"]
        )

        saved["Climatology_error_C"] = (
            saved["LST_C_observed"]
            - saved["Monthly_climatology_predicted_C"]
        )

        all_predictions.append(saved)

    if not all_metrics:
        raise ValueError("No evaluation results were generated.")

    # ==================================================
    # 5. SAVE AND PRINT RESULTS
    # ==================================================

    results_df = pd.DataFrame(all_metrics)

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("\nROLLING-ORIGIN RESULTS")
    print("-" * 100)

    print(results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print("\nFILES CREATED")
    print("-" * 65)
    print(RESULTS_FILE.name)
    print(PREDICTIONS_FILE.name)

    print("\nRolling-origin evaluation completed.")


if __name__ == "__main__":
    main()
