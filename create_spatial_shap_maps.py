
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_SHAP_Sample_Explanations.csv"
)

OUTPUT_FOLDER = FOLDER / "Bengaluru_UHI_Spatial_SHAP_Maps"

X_COLUMN = "utm_easting_m"
Y_COLUMN = "utm_northing_m"

MAPS = [
    ("LST_C_predicted", "Predicted Cropland LST (°C)",
     "Predicted_LST"),

    ("SHAP_doy_sin_C", "SHAP Contribution: Seasonal Sin (°C)",
     "SHAP_doy_sin"),

    ("SHAP_doy_cos_C", "SHAP Contribution: Seasonal Cos (°C)",
     "SHAP_doy_cos"),

    ("SHAP_NDMI_C", "SHAP Contribution: NDMI (°C)",
     "SHAP_NDMI"),

    ("SHAP_NDVI_C", "SHAP Contribution: NDVI (°C)",
     "SHAP_NDVI"),

    ("SHAP_builtup_fraction_500m_2021_C",
     "SHAP Contribution: Built-up Fraction (°C)",
     "SHAP_Builtup"),

    ("SHAP_elevation_m_C", "SHAP Contribution: Elevation (°C)",
     "SHAP_Elevation"),

    ("SHAP_year_C", "SHAP Contribution: Year (°C)",
     "SHAP_Year"),
]


# ==================================================
# 2. LOAD EXISTING SHAP RESULTS
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find: {INPUT_FILE.name}\n"
            "Run explain_baseline_shap.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    required_columns = [
        X_COLUMN,
        Y_COLUMN,
        "longitude",
        "latitude",
    ] + [item[0] for item in MAPS]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in SHAP file: {missing}"
        )

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        subset=required_columns
    ).copy()

    if data.empty:
        raise ValueError(
            "No complete spatial SHAP records found."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True, exist_ok=True
    )

    print("SPATIAL SHAP MAP GENERATION")
    print("-" * 55)
    print(f"Input records: {len(data):,}")
    print(f"Output folder: {OUTPUT_FOLDER}")

    # ==================================================
    # 3. GENERATE SPATIAL MAPS
    # ==================================================

    for column, title, filename in MAPS:

        values = data[column].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 7))

        if column == "LST_C_predicted":

            scatter = ax.scatter(
                data[X_COLUMN] / 1000,
                data[Y_COLUMN] / 1000,
                c=values,
                cmap="inferno",
                s=13,
                alpha=0.85,
            )

        else:

            # Use a balanced color range around zero.
            # Positive and negative SHAP contributions
            # are shown with different colors.

            limit = np.nanpercentile(
                np.abs(values), 98
            )

            if not np.isfinite(limit) or limit == 0:
                limit = 1.0

            scatter = ax.scatter(
                data[X_COLUMN] / 1000,
                data[Y_COLUMN] / 1000,
                c=values,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                s=13,
                alpha=0.85,
            )

        colorbar = fig.colorbar(
            scatter, ax=ax, shrink=0.8
        )

        if column == "LST_C_predicted":
            colorbar.set_label(
                "Predicted LST (°C)"
            )
        else:
            colorbar.set_label(
                "SHAP contribution (°C)"
            )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("UTM Easting (km)")
        ax.set_ylabel("UTM Northing (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

        fig.tight_layout()

        output_file = (
            OUTPUT_FOLDER / f"{filename}.png"
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Created: {output_file.name}")

    print("\nSPATIAL SHAP MAPS COMPLETED")
    print(f"Maps saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
