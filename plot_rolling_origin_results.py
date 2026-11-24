
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Rolling_Origin_Results.csv"
)

OUTPUT_FOLDER = FOLDER / "Bengaluru_UHI_Final_Result_Figures"


# ==================================================
# 2. LOAD AND VALIDATE RESULTS
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}"
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "test_year",
        "approach",
        "RMSE_C",
        "MAE_C",
        "R2",
        "mean_error_C",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True, exist_ok=True
    )

    approaches = [
        "Linear Regression",
        "Monthly climatology",
    ]

    data = data[
        data["approach"].isin(approaches)
    ].copy()

    data["test_year"] = pd.to_numeric(
        data["test_year"], errors="raise"
    )

    data = data.sort_values("test_year")


    # ==================================================
    # 3. PLOTTING FUNCTION
    # ==================================================

    def create_comparison_plot(
        metric,
        title,
        ylabel,
        filename,
        zero_line=False,
    ):

        pivot = data.pivot(
            index="test_year",
            columns="approach",
            values=metric,
        )

        pivot = pivot.reindex(columns=approaches)

        ax = pivot.plot(
            kind="bar",
            figsize=(9, 6),
            width=0.75,
        )

        if zero_line:
            ax.axhline(0, linestyle="--")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Test year")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(
            [str(int(year)) for year in pivot.index],
            rotation=0,
        )

        ax.legend(title="Approach")
        ax.grid(axis="y", alpha=0.25)

        fig = ax.get_figure()
        fig.tight_layout()

        output_file = OUTPUT_FOLDER / filename

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Created: {filename}")


    # ==================================================
    # 4. CREATE THREE COMPARISON FIGURES
    # ==================================================

    create_comparison_plot(
        metric="RMSE_C",
        title="Rolling-Origin Evaluation: RMSE",
        ylabel="RMSE (°C)",
        filename="Rolling_Origin_RMSE.png",
    )

    create_comparison_plot(
        metric="MAE_C",
        title="Rolling-Origin Evaluation: MAE",
        ylabel="MAE (°C)",
        filename="Rolling_Origin_MAE.png",
    )

    create_comparison_plot(
        metric="mean_error_C",
        title="Rolling-Origin Evaluation: Mean Prediction Bias",
        ylabel="Observed - Predicted LST (°C)",
        filename="Rolling_Origin_Mean_Bias.png",
        zero_line=True,
    )


    # ==================================================
    # 5. PRINT A SUMMARY
    # ==================================================

    print("\nBEST APPROACH BY YEAR")
    print("-" * 55)

    for year in sorted(data["test_year"].unique()):

        subset = data[data["test_year"] == year]

        best = subset.loc[
            subset["RMSE_C"].idxmin()
        ]

        print(
            f"{int(year)}: {best['approach']} "
            f"(RMSE = {best['RMSE_C']:.4f} °C)"
        )

    print("\nOUTPUT FOLDER")
    print(OUTPUT_FOLDER)

    print("\nResult figures created successfully.")


if __name__ == "__main__":
    main()
