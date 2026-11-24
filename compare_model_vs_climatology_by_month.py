
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. FILES
# ==================================================

FOLDER = Path(__file__).resolve().parent

SPLIT_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Split_2021_2025.csv"
)

PREDICTIONS_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Test_Predictions.csv"
)

OUTPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Monthly_Model_Comparison.csv"
)


# ==================================================
# 2. METRICS
# ==================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    return {
        "MAE_C": mean_absolute_error(actual, predicted),
        "RMSE_C": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "mean_error_C": errors.mean(),
    }


# ==================================================
# 3. LOAD DATA
# ==================================================

def main():

    for file in [SPLIT_FILE, PREDICTIONS_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"File not found: {file.name}"
            )

    history_data = pd.read_csv(SPLIT_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    history_data["date"] = pd.to_datetime(
        history_data["date"], errors="coerce"
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )

    # Use only predictions for the model selected on
    # 2024 validation data.
    predictions = predictions[
        (predictions["feature_set"] == "Full model")
        & (predictions["model"] == "Linear Regression")
    ].copy()

    if predictions.empty:
        raise ValueError(
            "Selected Linear Regression predictions not found."
        )

    for column in ["LST_C"]:
        history_data[column] = pd.to_numeric(
            history_data[column], errors="coerce"
        )

    for column in ["LST_C_observed", "LST_C_predicted"]:
        predictions[column] = pd.to_numeric(
            predictions[column], errors="coerce"
        )

    history_data = history_data.dropna(
        subset=["date", "LST_C", "temporal_split"]
    )

    predictions = predictions.dropna(
        subset=["date", "LST_C_observed", "LST_C_predicted"]
    ).copy()

    # Historical data includes 2021-2024 only.
    history = history_data[
        history_data["temporal_split"].isin(
            ["train", "validation"]
        )
    ].copy()

    if history["date"].dt.year.max() > 2024:
        raise ValueError(
            "The historical baseline contains test-year data."
        )

    predictions["month"] = predictions["date"].dt.month
    history["month"] = history["date"].dt.month

    # Historical mean LST for each calendar month.
    monthly_means = history.groupby("month")["LST_C"].mean()

    overall_historical_mean = history["LST_C"].mean()

    predictions["climatology_prediction_C"] = (
        predictions["month"]
        .map(monthly_means)
        .fillna(overall_historical_mean)
    )

    print("MONTHLY MODEL VS CLIMATOLOGY")
    print("-" * 70)
    print(f"Historical years: 2021-2024")
    print(f"Test year: 2025")
    print(f"Test samples: {len(predictions):,}")
    print(f"Test scenes: {predictions['product_id'].nunique()}")


    # ==================================================
    # 4. EVALUATE EACH TEST MONTH
    # ==================================================

    results = []

    for month, group in predictions.groupby("month"):

        actual = group["LST_C_observed"]

        model_metrics = calculate_metrics(
            actual,
            group["LST_C_predicted"],
        )

        climatology_metrics = calculate_metrics(
            actual,
            group["climatology_prediction_C"],
        )

        model_mae = model_metrics["MAE_C"]
        climatology_mae = climatology_metrics["MAE_C"]

        results.append({
            "month": int(month),
            "month_name": calendar.month_name[int(month)],
            "scenes": group["product_id"].nunique(),
            "samples": len(group),

            "model_MAE_C": model_mae,
            "climatology_MAE_C": climatology_mae,

            "model_RMSE_C": model_metrics["RMSE_C"],
            "climatology_RMSE_C": climatology_metrics["RMSE_C"],

            "model_mean_error_C": model_metrics["mean_error_C"],
            "climatology_mean_error_C": (
                climatology_metrics["mean_error_C"]
            ),

            "lower_MAE": (
                "Linear Regression"
                if model_mae < climatology_mae
                else "Monthly climatology"
            ),
        })

    result = pd.DataFrame(results).sort_values(
        "month"
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False)


    # ==================================================
    # 5. PRINT RESULTS
    # ==================================================

    print("\nRESULTS BY MONTH")
    print("-" * 110)

    print(result.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    ))

    model_wins = int(
        (result["lower_MAE"] == "Linear Regression").sum()
    )

    climatology_wins = int(
        (result["lower_MAE"] == "Monthly climatology").sum()
    )

    print("\nSUMMARY")
    print("-" * 70)
    print(f"Months where Linear Regression has lower MAE: {model_wins}")
    print(f"Months where climatology has lower MAE: {climatology_wins}")

    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
