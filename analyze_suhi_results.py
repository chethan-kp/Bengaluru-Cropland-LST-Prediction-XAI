
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_SUHI_Intensity_2021_2025.csv"

OUTPUT_FOLDER = FOLDER / "Bengaluru_SUHI_Analysis"

MIN_VALID_PIXELS = 1000


# ==================================================
# 2. LOAD AND VALIDATE THE EXPORTED DATA
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}.\n"
            "Download the SUHI CSV from Google Drive "
            "and place it in the Bengaluru_UHI folder."
        )

    data = pd.read_csv(INPUT_FILE)

    required = [
        "date",
        "product_id",
        "urban_mean_LST_C",
        "urban_valid_pixels",
        "rural_mean_LST_C",
        "rural_valid_pixels",
        "SUHI_intensity_C",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    numeric_columns = [
        "urban_mean_LST_C",
        "urban_valid_pixels",
        "rural_mean_LST_C",
        "rural_valid_pixels",
        "SUHI_intensity_C",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        subset=required
    ).copy()

    # Apply the same minimum valid-pixel threshold
    # used by the Earth Engine calculation.
    data = data[
        (data["urban_valid_pixels"] >= MIN_VALID_PIXELS)
        & (data["rural_valid_pixels"] >= MIN_VALID_PIXELS)
    ].copy()

    # Check the intensity calculation:
    # urban mean LST minus rural mean LST.
    calculated = (
        data["urban_mean_LST_C"]
        - data["rural_mean_LST_C"]
    )

    difference = (
        calculated - data["SUHI_intensity_C"]
    ).abs()

    if (difference > 0.02).any():
        raise ValueError(
            "Some SUHI values do not match "
            "urban LST minus rural LST."
        )

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid SUHI observations remain. "
            "Check the exported CSV and pixel thresholds."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True, exist_ok=True
    )

    print("SUHI RESULTS AUDIT")
    print("-" * 60)
    print(f"Valid acquisitions: {len(data)}")
    print(f"Unique product IDs: {data['product_id'].nunique()}")
    print(
        "SUHI calculation verified: "
        "urban mean LST - rural mean LST"
    )


    # ==================================================
    # 3. OVERALL RESULTS
    # ==================================================

    suhi = data["SUHI_intensity_C"]

    print("\nOVERALL SUHI SUMMARY")
    print("-" * 60)
    print(f"Mean intensity:   {suhi.mean():.3f} °C")
    print(f"Median intensity: {suhi.median():.3f} °C")
    print(f"Standard deviation: {suhi.std():.3f} °C")
    print(f"Minimum:          {suhi.min():.3f} °C")
    print(f"Maximum:          {suhi.max():.3f} °C")
    print(
        f"Positive intensity: {(suhi > 0).sum()} "
        f"acquisitions ({100 * (suhi > 0).mean():.1f}%)"
    )
    print(
        f"Negative intensity: {(suhi < 0).sum()} "
        f"acquisitions ({100 * (suhi < 0).mean():.1f}%)"
    )
    print(f"Near zero: {(suhi == 0).sum()} acquisitions")


    # ==================================================
    # 4. YEAR-WISE SUMMARY
    # ==================================================

    yearly = (
        data.groupby("year")
        .agg(
            acquisitions=("product_id", "nunique"),
            mean_SUHI_C=("SUHI_intensity_C", "mean"),
            median_SUHI_C=("SUHI_intensity_C", "median"),
            std_SUHI_C=("SUHI_intensity_C", "std"),
            minimum_SUHI_C=("SUHI_intensity_C", "min"),
            maximum_SUHI_C=("SUHI_intensity_C", "max"),
            positive_percent=(
                "SUHI_intensity_C",
                lambda x: 100 * (x > 0).mean()
            ),
        )
        .reset_index()
    )

    yearly_file = (
        OUTPUT_FOLDER / "SUHI_Summary_By_Year.csv"
    )

    yearly.to_csv(yearly_file, index=False)

    print("\nYEAR-WISE SUHI SUMMARY")
    print("-" * 80)
    print(
        yearly.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


    # ==================================================
    # 5. MONTH-WISE SUMMARY
    # ==================================================

    monthly = (
        data.groupby("month")
        .agg(
            acquisitions=("product_id", "nunique"),
            mean_SUHI_C=("SUHI_intensity_C", "mean"),
            median_SUHI_C=("SUHI_intensity_C", "median"),
            std_SUHI_C=("SUHI_intensity_C", "std"),
        )
        .reset_index()
    )

    monthly_file = (
        OUTPUT_FOLDER / "SUHI_Summary_By_Month.csv"
    )

    monthly.to_csv(monthly_file, index=False)

    print("\nMONTH-WISE SUHI SUMMARY")
    print("-" * 70)
    print(
        monthly.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


    # ==================================================
    # 6. SAVE VALIDATED PER-SCENE RESULTS
    # ==================================================

    validated_file = (
        OUTPUT_FOLDER / "SUHI_Validated_Per_Scene.csv"
    )

    data.to_csv(validated_file, index=False)


    # ==================================================
    # 7. PLOT SUHI INTENSITY BY ACQUISITION DATE
    # ==================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        data["date"],
        data["SUHI_intensity_C"],
        marker="o",
        linewidth=1.2,
    )

    ax.axhline(0, linestyle="--")

    ax.set_title(
        "Bengaluru Surface Urban–Rural Temperature Contrast"
    )
    ax.set_xlabel("Landsat acquisition date")
    ax.set_ylabel("Urban LST - Rural cropland LST (°C)")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    time_plot = (
        OUTPUT_FOLDER / "SUHI_Intensity_Time_Series.png"
    )

    fig.savefig(
        time_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ==================================================
    # 8. PLOT YEAR-WISE MEAN SUHI
    # ==================================================

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        yearly["year"].astype(str),
        yearly["mean_SUHI_C"],
    )

    ax.axhline(0, linestyle="--")

    ax.set_title("Mean SUHI Intensity by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean urban–rural LST difference (°C)")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    yearly_plot = (
        OUTPUT_FOLDER / "SUHI_Mean_By_Year.png"
    )

    fig.savefig(
        yearly_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ==================================================
    # 9. COMPLETION
    # ==================================================

    print("\nFILES CREATED")
    print("-" * 60)
    print(yearly_file.name)
    print(monthly_file.name)
    print(validated_file.name)
    print(time_plot.name)
    print(yearly_plot.name)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    print("\nSUHI analysis completed.")


if __name__ == "__main__":
    main()
