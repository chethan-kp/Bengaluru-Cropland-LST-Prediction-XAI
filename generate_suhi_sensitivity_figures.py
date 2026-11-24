
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent

SCENE_FILE = ROOT / "SUHI_Pixel_Coverage_By_Scene.csv"
BAND_FILE = ROOT / "SUHI_Pixel_Distance_Band_Summary.csv"

OUTPUT_DIR = ROOT / "Bengaluru_SUHI_Sensitivity_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCES = {
    "10-20 km": ["10-15 km", "15-20 km"],
    "15-25 km": ["15-20 km", "20-25 km"],
    "20-30 km": ["20-25 km", "25-30 km"],
    "10-30 km sample": [
        "10-15 km", "15-20 km",
        "20-25 km", "25-30 km"
    ]
}


# =====================================================
# 2. LOAD INPUT FILES
# =====================================================

def load_inputs():

    if not SCENE_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {SCENE_FILE.name}"
        )

    if not BAND_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {BAND_FILE.name}"
        )

    scenes = pd.read_csv(SCENE_FILE)
    bands = pd.read_csv(BAND_FILE)

    scene_columns = [
        "product_id", "year",
        "urban_mean_LST_C", "SUHI_intensity_C"
    ]

    band_columns = [
        "product_id", "year",
        "distance_band", "pixel_count", "mean_LST_C"
    ]

    for col in scene_columns:
        if col not in scenes.columns:
            raise ValueError(
                f"Missing scene column: {col}"
            )

    for col in band_columns:
        if col not in bands.columns:
            raise ValueError(
                f"Missing distance-band column: {col}"
            )

    scenes["year"] = pd.to_numeric(
        scenes["year"], errors="coerce"
    )

    scenes["urban_mean_LST_C"] = pd.to_numeric(
        scenes["urban_mean_LST_C"], errors="coerce"
    )

    scenes["SUHI_intensity_C"] = pd.to_numeric(
        scenes["SUHI_intensity_C"], errors="coerce"
    )

    bands["year"] = pd.to_numeric(
        bands["year"], errors="coerce"
    )

    bands["pixel_count"] = pd.to_numeric(
        bands["pixel_count"], errors="coerce"
    )

    bands["mean_LST_C"] = pd.to_numeric(
        bands["mean_LST_C"], errors="coerce"
    )

    scenes = scenes.dropna(
        subset=[
            "product_id", "year",
            "urban_mean_LST_C",
            "SUHI_intensity_C"
        ]
    )

    bands = bands.dropna(
        subset=[
            "product_id", "year",
            "distance_band",
            "pixel_count", "mean_LST_C"
        ]
    )

    scenes = scenes.drop_duplicates(
        subset=["product_id"]
    )

    print("Input files loaded successfully.")

    return scenes, bands


# =====================================================
# 3. CALCULATE DISTANCE-BAND SENSITIVITY
# =====================================================

def calculate_results(scenes, bands):

    results = []

    for reference, required_bands in REFERENCES.items():

        selected = bands[
            bands["distance_band"].isin(required_bands)
        ].copy()

        selected["weighted_lst"] = (
            selected["pixel_count"]
            * selected["mean_LST_C"]
        )

        grouped = (
            selected.groupby("product_id")
            .agg(
                bands_present=("distance_band", "nunique"),
                distance_pixel_count=("pixel_count", "sum"),
                weighted_lst_sum=("weighted_lst", "sum")
            )
            .reset_index()
        )

        # Require every distance band for the scene
        grouped = grouped[
            grouped["bands_present"] == len(required_bands)
        ].copy()

        grouped["sampled_rural_mean_LST_C"] = (
            grouped["weighted_lst_sum"]
            / grouped["distance_pixel_count"]
        )

        # Use a separate column name to avoid duplicate
        # sample_pixel_count columns during the merge.
        grouped = grouped[
            [
                "product_id",
                "distance_pixel_count",
                "sampled_rural_mean_LST_C"
            ]
        ]

        merged = scenes.merge(
            grouped,
            on="product_id",
            how="inner",
            validate="one_to_one"
        )

        merged["reference"] = reference

        merged["sampled_SUHI_C"] = (
            merged["urban_mean_LST_C"]
            - merged["sampled_rural_mean_LST_C"]
        )

        merged["difference_vs_official_C"] = (
            merged["sampled_SUHI_C"]
            - merged["SUHI_intensity_C"]
        )

        results.append(merged)

        print(
            f"{reference}: "
            f"{merged['product_id'].nunique()} matched acquisitions"
        )

    if not results:
        raise ValueError("No sensitivity results were generated.")

    return pd.concat(results, ignore_index=True)


# =====================================================
# 4. CREATE SUMMARY TABLES
# =====================================================

def create_tables(per_scene):

    order = list(REFERENCES.keys())

    summary = (
        per_scene.groupby("reference")
        .agg(
            n_scenes=("product_id", "nunique"),

            mean_rural_LST_C=(
                "sampled_rural_mean_LST_C", "mean"
            ),

            mean_SUHI_C=(
                "sampled_SUHI_C", "mean"
            ),

            median_SUHI_C=(
                "sampled_SUHI_C", "median"
            ),

            sd_SUHI_C=(
                "sampled_SUHI_C", "std"
            ),

            positive_scenes=(
                "sampled_SUHI_C",
                lambda x: int((x > 0).sum())
            ),

            negative_scenes=(
                "sampled_SUHI_C",
                lambda x: int((x < 0).sum())
            ),

            mean_abs_difference_vs_official_C=(
                "difference_vs_official_C",
                lambda x: float(x.abs().mean())
            ),

            median_distance_pixel_count=(
                "distance_pixel_count", "median"
            )
        )
        .reindex(order)
        .reset_index()
    )

    annual = (
        per_scene.groupby(["reference", "year"])
        .agg(
            n_scenes=("product_id", "nunique"),

            mean_SUHI_C=(
                "sampled_SUHI_C", "mean"
            ),

            positive_scenes=(
                "sampled_SUHI_C",
                lambda x: int((x > 0).sum())
            ),

            negative_scenes=(
                "sampled_SUHI_C",
                lambda x: int((x < 0).sum())
            )
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "Sensitivity_Summary.csv",
        index=False
    )

    annual.to_csv(
        OUTPUT_DIR / "Sensitivity_By_Year.csv",
        index=False
    )

    per_scene.to_csv(
        OUTPUT_DIR / "Sensitivity_Per_Scene.csv",
        index=False
    )

    return summary, annual


# =====================================================
# 5. MAIN COMPARISON FIGURE
# =====================================================

def create_comparison_figure(summary, annual):

    order = list(REFERENCES.keys())

    summary = (
        summary.set_index("reference")
        .loc[order]
        .reset_index()
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6.5)
    )

    # Left plot: mean thermal contrast

    ax = axes[0]

    y = np.arange(len(summary))

    means = summary["mean_SUHI_C"].to_numpy()

    errors = (
        summary["sd_SUHI_C"]
        .fillna(0)
        .to_numpy()
    )

    ax.barh(
        y,
        means,
        xerr=errors,
        capsize=4
    )

    ax.set_yticks(y)
    ax.set_yticklabels(summary["reference"])

    ax.invert_yaxis()

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "Urban minus sampled rural LST (°C)"
    )

    ax.set_title(
        "(a) Mean contrast by rural reference band"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    for i, value in enumerate(means):

        ax.text(
            value - 0.04,
            i,
            f"{value:.2f}°C",
            ha="right",
            va="center",
            fontsize=9
        )

    # Right plot: yearly comparison

    ax = axes[1]

    for reference in order:

        temp = annual[
            annual["reference"] == reference
        ].sort_values("year")

        ax.plot(
            temp["year"],
            temp["mean_SUHI_C"],
            marker="o",
            linewidth=2,
            label=reference
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("Year")

    ax.set_ylabel("Mean contrast (°C)")

    ax.set_title("(b) Year-wise contrast")

    ax.set_xticks(
        sorted(annual["year"].dropna().unique())
    )

    ax.grid(alpha=0.25)

    ax.legend(
        title="Reference band",
        fontsize=8
    )

    fig.suptitle(
        "Bengaluru: Rural Reference Distance Sensitivity",
        fontsize=14,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.015,
        "Sample-based comparison on matched acquisitions; "
        "not a complete re-extraction of land-cover zones. "
        "Bars show mean ± 1 SD.",
        ha="center",
        fontsize=8
    )

    fig.tight_layout(
        rect=[0, 0.05, 1, 0.93]
    )

    fig.savefig(
        OUTPUT_DIR / "Sensitivity_Comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_DIR / "Sensitivity_Comparison.svg",
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# 6. ANNUAL TREND FIGURE
# =====================================================

def create_annual_figure(annual):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for reference in REFERENCES:

        temp = annual[
            annual["reference"] == reference
        ].sort_values("year")

        ax.plot(
            temp["year"],
            temp["mean_SUHI_C"],
            marker="o",
            linewidth=2,
            label=reference
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Annual Urban–Rural Surface Thermal Contrast"
    )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Urban minus sampled rural LST (°C)"
    )

    ax.set_xticks(
        sorted(annual["year"].dropna().unique())
    )

    ax.grid(alpha=0.25)

    ax.legend(title="Rural reference band")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "Sensitivity_Annual_Trend.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# 7. SCENE DISTRIBUTION FIGURE
# =====================================================

def create_distribution_figure(per_scene):

    order = list(REFERENCES.keys())

    values = [
        per_scene.loc[
            per_scene["reference"] == reference,
            "sampled_SUHI_C"
        ].dropna().to_numpy()

        for reference in order
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Do not use labels= here.
    # This works with the current Matplotlib API.
    ax.boxplot(
        values,
        showmeans=True
    )

    # Set tick positions and labels separately.
    ax.set_xticks(
        np.arange(1, len(order) + 1)
    )

    ax.set_xticklabels(order)

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Scene-level Distribution of Thermal Contrast"
    )

    ax.set_xlabel(
        "Rural reference distance band"
    )

    ax.set_ylabel(
        "Urban minus sampled rural LST (°C)"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=10
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "Sensitivity_Scene_Distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# 8. MAIN EXECUTION
# =====================================================

def main():

    print("\nBENGALURU SUHI SENSITIVITY FIGURE GENERATOR")
    print("=" * 55)

    scenes, bands = load_inputs()

    print("Calculating alternative reference bands...\n")

    per_scene = calculate_results(
        scenes, bands
    )

    summary, annual = create_tables(
        per_scene
    )

    print("\nGenerating figures...")

    create_comparison_figure(
        summary, annual
    )

    create_annual_figure(annual)

    create_distribution_figure(
        per_scene
    )

    print("\nSUMMARY RESULTS")

    print(
        summary[
            [
                "reference",
                "n_scenes",
                "mean_rural_LST_C",
                "mean_SUHI_C",
                "mean_abs_difference_vs_official_C"
            ]
        ].to_string(index=False)
    )

    print("\nALL FIGURES AND TABLES SAVED SUCCESSFULLY")

    print(f"\nOutput folder:\n{OUTPUT_DIR}")

    print("\nGenerated figures:")
    print("- Sensitivity_Comparison.png")
    print("- Sensitivity_Comparison.svg")
    print("- Sensitivity_Annual_Trend.png")
    print("- Sensitivity_Scene_Distribution.png")

    print("\nGenerated CSV files:")
    print("- Sensitivity_Summary.csv")
    print("- Sensitivity_By_Year.csv")
    print("- Sensitivity_Per_Scene.csv")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR OCCURRED:")
        print(error)
        raise
