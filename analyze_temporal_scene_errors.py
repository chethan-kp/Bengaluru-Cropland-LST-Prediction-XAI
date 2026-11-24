
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Test_Predictions.csv"
)

OUTPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Temporal_Error_By_Scene.csv"
)


# ==================================================
# 2. LOAD SELECTED TEMPORAL MODEL PREDICTIONS
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}. "
            "Run run_temporal_holdout.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "date",
        "product_id",
        "feature_set",
        "model",
        "LST_C_observed",
        "LST_C_predicted",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Analyze the configuration selected using 2024 validation.
    data = data[
        (data["feature_set"] == "Full model")
        & (data["model"] == "Linear Regression")
    ].copy()

    if data.empty:
        raise ValueError(
            "Selected-model predictions were not found."
        )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    for column in [
        "LST_C_observed",
        "LST_C_predicted",
    ]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        subset=required
    ).copy()

    if data.empty:
        raise ValueError("No valid predictions remain.")

    # Positive residual = underprediction.
    # Negative residual = overprediction.
    data["residual_C"] = (
        data["LST_C_observed"]
        - data["LST_C_predicted"]
    )

    data["absolute_error_C"] = (
        data["residual_C"].abs()
    )

    data["squared_error_C"] = (
        data["residual_C"] ** 2
    )


    # ==================================================
    # 3. CALCULATE METRICS BY ACQUISITION
    # ==================================================

    scenes = (
        data.groupby(
            ["date", "product_id"],
            as_index=False,
        )
        .agg(
            samples=("LST_C_observed", "size"),
            observed_mean_C=("LST_C_observed", "mean"),
            predicted_mean_C=("LST_C_predicted", "mean"),
            mean_error_C=("residual_C", "mean"),
            MAE_C=("absolute_error_C", "mean"),
            mean_squared_error_C=("squared_error_C", "mean"),
        )
    )

    scenes["RMSE_C"] = np.sqrt(
        scenes["mean_squared_error_C"]
    )

    scenes = scenes.drop(
        columns=["mean_squared_error_C"]
    )

    scenes["error_direction"] = np.select(
        [
            scenes["mean_error_C"] > 0,
            scenes["mean_error_C"] < 0,
        ],
        [
            "Underprediction",
            "Overprediction",
        ],
        default="Zero average bias",
    )

    scenes = scenes.sort_values(
        "MAE_C", ascending=False
    ).reset_index(drop=True)


    # ==================================================
    # 4. PRINT RESULTS
    # ==================================================

    print("TEMPORAL TEST ERROR BY SCENE")
    print("-" * 70)
    print(f"Test rows: {len(data):,}")
    print(f"Test scenes: {scenes['product_id'].nunique()}")

    print("\nSCENES SORTED BY MAE")
    print("-" * 90)

    display = scenes.copy()
    display["date"] = display["date"].dt.strftime(
        "%Y-%m-%d"
    )

    print(
        display[
            [
                "date",
                "samples",
                "observed_mean_C",
                "predicted_mean_C",
                "mean_error_C",
                "MAE_C",
                "RMSE_C",
                "error_direction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


    # ==================================================
    # 5. SAVE TABLE
    # ==================================================

    scenes["date"] = scenes["date"].dt.strftime(
        "%Y-%m-%d"
    )

    scenes.to_csv(
        OUTPUT_FILE, index=False
    )


    # ==================================================
    # 6. PLOT SCENE-LEVEL BIAS
    # ==================================================

    plot_data = scenes.sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        plot_data["date"],
        plot_data["mean_error_C"],
    )

    ax.axhline(0, linestyle="--")

    ax.set_title("2025 Temporal Test: Mean Error by Scene")
    ax.set_xlabel("Acquisition date")
    ax.set_ylabel("Observed - Predicted LST (°C)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    bias_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Scene_Bias.png"
    )

    fig.savefig(
        bias_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ==================================================
    # 7. PLOT MAE BY SCENE
    # ==================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        plot_data["date"],
        plot_data["MAE_C"],
    )

    ax.set_title("2025 Temporal Test: MAE by Scene")
    ax.set_xlabel("Acquisition date")
    ax.set_ylabel("Mean Absolute Error (°C)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    mae_file = (
        FOLDER / "Bengaluru_UHI_Temporal_Scene_MAE.png"
    )

    fig.savefig(
        mae_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ==================================================
    # 8. COMPLETION
    # ==================================================

    print("\nFILES CREATED")
    print("-" * 70)
    print(OUTPUT_FILE.name)
    print(bias_file.name)
    print(mae_file.name)

    print("\nTemporal scene error analysis completed.")


if __name__ == "__main__":
    main()
