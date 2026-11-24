
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

SPLIT_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Split_2021_2025.csv"
)

PREDICTIONS_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Test_Predictions.csv"
)

OUTPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Baseline_Comparison.csv"
)


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
        "MAE_C": mean_absolute_error(
            actual, predicted
        ),
        "R2": r2_score(actual, predicted),
        "mean_error_C": np.mean(
            actual - predicted
        ),
    }


# ==================================================
# 3. LOAD DATA
# ==================================================

def main():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {SPLIT_FILE.name}"
        )

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {PREDICTIONS_FILE.name}"
        )

    data = pd.read_csv(SPLIT_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required_data = ["date", "LST_C", "temporal_split"]

    required_predictions = [
        "date",
        "product_id",
        "feature_set",
        "model",
        "LST_C_observed",
        "LST_C_predicted",
    ]

    for frame, required, name in [
        (data, required_data, "temporal split"),
        (predictions, required_predictions, "predictions"),
    ]:
        missing = [c for c in required if c not in frame.columns]

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {missing}"
            )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )

    data["LST_C"] = pd.to_numeric(
        data["LST_C"], errors="coerce"
    )

    predictions["LST_C_observed"] = pd.to_numeric(
        predictions["LST_C_observed"], errors="coerce"
    )

    predictions["LST_C_predicted"] = pd.to_numeric(
        predictions["LST_C_predicted"], errors="coerce"
    )

    data = data.dropna(
        subset=required_data
    ).copy()

    predictions = predictions.dropna(
        subset=required_predictions
    ).copy()

    # Only historical observations available before the 2025 test.
    history = data[
        data["temporal_split"].isin(["train", "validation"])
    ].copy()

    if history["date"].dt.year.max() > 2024:
        raise ValueError(
            "Historical baseline contains data after 2024."
        )

    test_predictions = predictions[
        (predictions["feature_set"] == "Full model")
        & (predictions["model"] == "Linear Regression")
    ].copy()

    if test_predictions.empty:
        raise ValueError(
            "Selected model predictions were not found."
        )

    if test_predictions["date"].dt.year.nunique() != 1 or (
        test_predictions["date"].dt.year.iloc[0] != 2025
    ):
        raise ValueError(
            "Expected predictions for the 2025 test year."
        )

    test_predictions["month"] = (
        test_predictions["date"].dt.month
    )


    # ==================================================
    # 4. CALCULATE HISTORICAL BASELINES
    # ==================================================

    historical_mean = history["LST_C"].mean()

    monthly_means = (
        history.assign(
            month=history["date"].dt.month
        )
        .groupby("month")["LST_C"]
        .mean()
    )

    test_predictions["historical_mean_prediction_C"] = (
        historical_mean
    )

    # Use the historical monthly mean for each month.
    # If a month is missing from history, fall back to
    # the overall historical mean.
    test_predictions["monthly_climatology_prediction_C"] = (
        test_predictions["month"]
        .map(monthly_means)
        .fillna(historical_mean)
    )

    # ==================================================
    # 5. EVALUATE ALL THREE APPROACHES
    # ==================================================

    observed = test_predictions["LST_C_observed"]

    results = []

    approaches = {
        "Historical mean": (
            test_predictions["historical_mean_prediction_C"]
        ),
        "Monthly climatology": (
            test_predictions["monthly_climatology_prediction_C"]
        ),
        "Selected Linear Regression": (
            test_predictions["LST_C_predicted"]
        ),
    }

    for name, predicted in approaches.items():

        results.append({
            "approach": name,
            **calculate_metrics(observed, predicted),
        })

    result_df = pd.DataFrame(results).sort_values(
        "RMSE_C"
    ).reset_index(drop=True)


    # ==================================================
    # 6. PRINT AND SAVE RESULTS
    # ==================================================

    print("TEMPORAL BASELINE COMPARISON")
    print("-" * 65)

    print(f"Historical period: 2021–2024")
    print(f"Test period: 2025")
    print(f"Historical mean LST: {historical_mean:.4f} °C")
    print(f"Test samples: {len(test_predictions):,}")
    print(
        f"Test scenes: "
        f"{test_predictions['product_id'].nunique()}"
    )

    print("\nRESULTS ON 2025 TEST DATA")
    print("-" * 65)

    print(result_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    result_df.to_csv(
        OUTPUT_FILE, index=False
    )

    print(f"\nSaved: {OUTPUT_FILE.name}")
    print("\nBaseline comparison completed.")


if __name__ == "__main__":
    main()
