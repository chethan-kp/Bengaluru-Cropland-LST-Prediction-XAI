
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_UHI_Test_Predictions.csv"

OBSERVED = "LST_C_observed"
PREDICTED = "LST_C_predicted"


# ==================================================
# 2. METRIC FUNCTION
# ==================================================

def calculate_metrics(group):

    actual = group[OBSERVED].to_numpy()
    predicted = group[PREDICTED].to_numpy()

    errors = actual - predicted

    return {
        "samples": len(group),
        "MAE_C": mean_absolute_error(actual, predicted),
        "RMSE_C": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted) if len(group) >= 2 else np.nan,
        "mean_error_C": errors.mean(),
    }


# ==================================================
# 3. LOAD TEST PREDICTIONS
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}. "
            "Run train_baseline_models.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    required = [OBSERVED, PREDICTED, "year"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in [OBSERVED, PREDICTED, "year"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[OBSERVED, PREDICTED, "year"]).copy()

    if data.empty:
        raise ValueError("No valid test predictions remain.")

    # Recalculate errors directly from observed and predicted values.
    # Positive residual = underprediction.
    # Negative residual = overprediction.

    data["residual_C"] = data[OBSERVED] - data[PREDICTED]
    data["absolute_error_C"] = data["residual_C"].abs()

    if "month" not in data.columns:
        if "date" not in data.columns:
            raise ValueError("Neither month nor date column exists.")

        data["month"] = pd.to_datetime(
            data["date"], errors="coerce"
        ).dt.month
    else:
        data["month"] = pd.to_numeric(
            data["month"], errors="coerce"
        )

    print("TEST ERROR ANALYSIS")
    print("-" * 60)
    print(f"Test samples analyzed: {len(data):,}")

    # ==================================================
    # 4. OVERALL PERFORMANCE
    # ==================================================

    overall = calculate_metrics(data)

    print("\nOVERALL TEST PERFORMANCE")
    print("-" * 60)

    for metric, value in overall.items():
        if metric == "samples":
            print(f"{metric}: {value:,}")
        else:
            print(f"{metric}: {value:.4f}")

    # ==================================================
    # 5. PERFORMANCE BY YEAR
    # ==================================================

    yearly_rows = []

    for year, group in data.groupby("year"):
        yearly_rows.append({
            "year": int(year),
            **calculate_metrics(group),
        })

    yearly = pd.DataFrame(yearly_rows).sort_values("year")

    yearly_file = FOLDER / "Bengaluru_UHI_Error_Metrics_By_Year.csv"
    yearly.to_csv(yearly_file, index=False)

    print("\nERROR METRICS BY YEAR")
    print("-" * 60)
    print(yearly.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    # ==================================================
    # 6. PERFORMANCE BY MONTH
    # ==================================================

    valid_months = data[
        data["month"].between(1, 12)
    ].copy()

    monthly_rows = []

    for month, group in valid_months.groupby("month"):
        monthly_rows.append({
            "month": int(month),
            "month_name": calendar.month_name[int(month)],
            **calculate_metrics(group),
        })

    monthly = pd.DataFrame(monthly_rows).sort_values("month")

    monthly_file = FOLDER / "Bengaluru_UHI_Error_Metrics_By_Month.csv"
    monthly.to_csv(monthly_file, index=False)

    print("\nERROR METRICS BY MONTH")
    print("-" * 60)

    if not monthly.empty:
        print(monthly.to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))
    else:
        print("No valid month values available.")

    # ==================================================
    # 7. PLOT: OBSERVED VS PREDICTED LST
    # ==================================================

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        data[OBSERVED],
        data[PREDICTED],
        s=8,
        alpha=0.15,
    )

    minimum = min(data[OBSERVED].min(), data[PREDICTED].min())
    maximum = max(data[OBSERVED].max(), data[PREDICTED].max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    ax.set_xlabel("Observed LST (°C)")
    ax.set_ylabel("Predicted LST (°C)")
    ax.set_title("Observed vs Predicted Cropland LST")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        FOLDER / "Bengaluru_UHI_Observed_vs_Predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ==================================================
    # 8. PLOT: RESIDUAL DISTRIBUTION
    # ==================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(data["residual_C"], bins=50)
    ax.axvline(0, linestyle="--")

    ax.set_xlabel("Observed - Predicted LST (°C)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of Test Prediction Errors")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        FOLDER / "Bengaluru_UHI_Residual_Histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ==================================================
    # 9. PLOT: RESIDUALS VS PREDICTED LST
    # ==================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        data[PREDICTED],
        data["residual_C"],
        s=8,
        alpha=0.15,
    )

    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted LST (°C)")
    ax.set_ylabel("Observed - Predicted (°C)")
    ax.set_title("Residuals vs Predicted LST")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        FOLDER / "Bengaluru_UHI_Residuals_vs_Predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ==================================================
    # 10. PLOT: MONTHLY MAE
    # ==================================================

    if not monthly.empty:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            monthly["month_name"],
            monthly["MAE_C"],
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Absolute Error (°C)")
        ax.set_title("Test Prediction Error by Month")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            FOLDER / "Bengaluru_UHI_Monthly_MAE.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ==================================================
    # 11. COMPLETION
    # ==================================================

    print("\nFILES CREATED")
    print("-" * 60)

    files = [
        yearly_file,
        monthly_file,
        FOLDER / "Bengaluru_UHI_Observed_vs_Predicted.png",
        FOLDER / "Bengaluru_UHI_Residual_Histogram.png",
        FOLDER / "Bengaluru_UHI_Residuals_vs_Predicted.png",
    ]

    if not monthly.empty:
        files.append(FOLDER / "Bengaluru_UHI_Monthly_MAE.png")

    for file in files:
        print(file.name)

    print("\nError analysis completed successfully.")


if __name__ == "__main__":
    main()
