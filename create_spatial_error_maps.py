
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_UHI_Test_Predictions.csv"

OUTPUT_FOLDER = FOLDER / "Bengaluru_UHI_Spatial_Error_Maps"

X = "utm_easting_m"
Y = "utm_northing_m"

OBSERVED = "LST_C_observed"
PREDICTED = "LST_C_predicted"


# ==================================================
# 2. LOAD AND VALIDATE
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}"
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        X, Y, OBSERVED, PREDICTED,
        "block_x", "block_y"
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(subset=required).copy()

    if data.empty:
        raise ValueError("No valid test predictions found.")

    # Residual = observed - predicted
    # Negative: model overpredicted temperature
    # Positive: model underpredicted temperature

    data["residual_C"] = (
        data[OBSERVED] - data[PREDICTED]
    )

    data["absolute_error_C"] = data["residual_C"].abs()

    data["squared_error_C"] = data["residual_C"] ** 2

    OUTPUT_FOLDER.mkdir(
        parents=True, exist_ok=True
    )

    print("SPATIAL ERROR ANALYSIS")
    print("-" * 55)
    print(f"Test observations: {len(data):,}")

    # ==================================================
    # 3. AGGREGATE BY HELD-OUT SPATIAL BLOCK
    # ==================================================

    block_metrics = (
        data.groupby(["block_x", "block_y"], as_index=False)
        .agg(
            easting_m=(X, "mean"),
            northing_m=(Y, "mean"),
            samples=(OBSERVED, "size"),
            mean_observed_LST_C=(OBSERVED, "mean"),
            mean_predicted_LST_C=(PREDICTED, "mean"),
            mean_residual_C=("residual_C", "mean"),
            MAE_C=("absolute_error_C", "mean"),
            mean_squared_error_C=("squared_error_C", "mean"),
        )
    )

    block_metrics["RMSE_C"] = np.sqrt(
        block_metrics["mean_squared_error_C"]
    )

    block_metrics = block_metrics.drop(
        columns=["mean_squared_error_C"]
    )

    block_metrics["error_direction"] = np.select(
        [
            block_metrics["mean_residual_C"] > 0,
            block_metrics["mean_residual_C"] < 0,
        ],
        [
            "Underprediction",
            "Overprediction",
        ],
        default="Near-zero average bias",
    )

    output_csv = (
        OUTPUT_FOLDER / "Spatial_Error_Metrics_By_Block.csv"
    )

    block_metrics.to_csv(
        output_csv, index=False
    )

    print(f"Spatial blocks analyzed: {len(block_metrics)}")

    print("\nBLOCK ERROR SUMMARY")
    print("-" * 55)

    print(
        block_metrics[
            [
                "block_x", "block_y", "samples",
                "mean_residual_C", "MAE_C", "RMSE_C"
            ]
        ]
        .sort_values("MAE_C", ascending=False)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        )
    )

    # ==================================================
    # 4. FUNCTION TO CREATE BLOCK MAPS
    # ==================================================

    def make_map(
        value_column,
        title,
        colorbar_label,
        filename,
        diverging=False,
    ):

        values = block_metrics[value_column].to_numpy()

        fig, ax = plt.subplots(figsize=(9, 7))

        if diverging:

            limit = np.nanpercentile(
                np.abs(values), 98
            )

            if not np.isfinite(limit) or limit == 0:
                limit = 1.0

            scatter = ax.scatter(
                block_metrics["easting_m"] / 1000,
                block_metrics["northing_m"] / 1000,
                c=values,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                s=150,
                marker="s",
                edgecolors="black",
                linewidths=0.35,
            )

        else:

            scatter = ax.scatter(
                block_metrics["easting_m"] / 1000,
                block_metrics["northing_m"] / 1000,
                c=values,
                cmap="inferno",
                s=150,
                marker="s",
                edgecolors="black",
                linewidths=0.35,
            )

        colorbar = fig.colorbar(
            scatter, ax=ax, shrink=0.8
        )

        colorbar.set_label(colorbar_label)

        ax.set_title(title)
        ax.set_xlabel("UTM Easting (km)")
        ax.set_ylabel("UTM Northing (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

        fig.tight_layout()

        filepath = OUTPUT_FOLDER / filename

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Created: {filename}")

    # ==================================================
    # 5. MAP A: MEAN SIGNED ERROR
    # ==================================================

    make_map(
        value_column="mean_residual_C",
        title="Mean Prediction Error by Spatial Block",
        colorbar_label="Observed - Predicted LST (°C)",
        filename="Spatial_Mean_Residual_By_Block.png",
        diverging=True,
    )

    # ==================================================
    # 6. MAP B: MEAN ABSOLUTE ERROR
    # ==================================================

    make_map(
        value_column="MAE_C",
        title="Mean Absolute Prediction Error by Block",
        colorbar_label="MAE (°C)",
        filename="Spatial_MAE_By_Block.png",
    )

    # ==================================================
    # 7. MAP C: ROOT MEAN SQUARED ERROR
    # ==================================================

    make_map(
        value_column="RMSE_C",
        title="Root Mean Squared Prediction Error by Block",
        colorbar_label="RMSE (°C)",
        filename="Spatial_RMSE_By_Block.png",
    )

    print("\nSPATIAL ERROR MAPS COMPLETED")
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
