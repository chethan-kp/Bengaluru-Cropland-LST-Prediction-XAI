
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Scene_Test_Predictions.csv"
)

OUTPUT_CSV = (
    FOLDER / "Bengaluru_UHI_Scene_Error_By_Acquisition.csv"
)


# ==================================================
# 2. LOAD AND VALIDATE
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}. "
            "Run train_scene_heldout_models.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "date",
        "product_id",
        "year",
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

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    for column in [
        "year",
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
        raise ValueError("No valid test predictions found.")

    # Positive error means underprediction.
    # Negative error means overprediction.

    data["mean_error_C"] = (
        data["LST_C_observed"]
        - data["LST_C_predicted"]
    )

    data["absolute_error_C"] = (
        data["mean_error_C"].abs()
    )

    data["squared_error_C"] = (
        data["mean_error_C"] ** 2
    )

    print("SCENE-HELD-OUT ERROR ANALYSIS")
    print("-" * 60)
    print(f"Test rows: {len(data):,}")
    print(f"Test scenes: {data['product_id'].nunique()}")


    # ==================================================
    # 3. CALCULATE METRICS FOR EACH ACQUISITION
    # ==================================================

    scene_metrics = (
        data.groupby(
            ["date", "year", "product_id"],
            as_index=False,
        )
        .agg(
            samples=("LST_C_observed", "size"),
            observed_mean_C=("LST_C_observed", "mean"),
            predicted_mean_C=("LST_C_predicted", "mean"),
            mean_error_C=("mean_error_C", "mean"),
            MAE_C=("absolute_error_C", "mean"),
            mean_squared_error_C=("squared_error_C", "mean"),
        )
    )

    scene_metrics["RMSE_C"] = np.sqrt(
        scene_metrics["mean_squared_error_C"]
    )

    scene_metrics["error_direction"] = np.select(
        [
            scene_metrics["mean_error_C"] > 0,
            scene_metrics["mean_error_C"] < 0,
        ],
        [
            "Underprediction",
            "Overprediction",
        ],
        default="Zero average bias",
    )

    scene_metrics = scene_metrics.drop(
        columns=["mean_squared_error_C"]
    )

    scene_metrics["date"] = (
        scene_metrics["date"].dt.strftime("%Y-%m-%d")
    )

    scene_metrics = scene_metrics.sort_values(
        "date"
    ).reset_index(drop=True)

    scene_metrics.to_csv(
        OUTPUT_CSV, index=False
    )

    print("\nMETRICS BY ACQUISITION")
    print("-" * 80)

    print(
        scene_metrics[
            [
                "date",
                "product_id",
                "samples",
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
    # 4. IDENTIFY LARGEST ERRORS AND BIASES
    # ==================================================

    highest_error = scene_metrics.sort_values(
        "MAE_C", ascending=False
    ).head(3)

    print("\nTHREE HIGHEST-ERROR SCENES")
    print("-" * 60)

    print(
        highest_error[
            ["date", "samples", "MAE_C", "RMSE_C"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nTHREE MOST UNDERPREDICTED SCENES")
    print("-" * 60)

    print(
        scene_metrics.nlargest(
            3, "mean_error_C"
        )[
            ["date", "samples", "mean_error_C", "MAE_C"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


    # ==================================================
    # 5. PLOT: MEAN ERROR BY SCENE
    # ==================================================

    labels = scene_metrics["date"].tolist()
    x_positions = np.arange(len(scene_metrics))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x_positions,
        scene_metrics["mean_error_C"],
    )

    ax.axhline(0, linestyle="--")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Acquisition date")
    ax.set_ylabel("Observed - Predicted LST (°C)")
    ax.set_title("Mean Prediction Bias by Held-Out Scene")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    bias_plot = (
        FOLDER / "Bengaluru_UHI_Scene_Error_Bias.png"
    )

    fig.savefig(
        bias_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ==================================================
    # 6. PLOT: MAE BY SCENE
    # ==================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x_positions,
        scene_metrics["MAE_C"],
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Acquisition date")
    ax.set_ylabel("Mean Absolute Error (°C)")
    ax.set_title("Prediction Error by Held-Out Scene")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    mae_plot = (
        FOLDER / "Bengaluru_UHI_Scene_Error_MAE.png"
    )

    fig.savefig(
        mae_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ==================================================
    # 7. COMPLETION
    # ==================================================

    print("\nFILES CREATED")
    print("-" * 60)
    print(OUTPUT_CSV.name)
    print(bias_plot.name)
    print(mae_plot.name)

    print("\nScene-level error analysis completed.")


if __name__ == "__main__":
    main()
