import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

raster_path = Path(
    "Bengaluru_2026_Prediction/"
    "Bengaluru_2026_Cropland_Predictors_30m.tif"
)

csv_path = Path("Bengaluru_UHI_Deduplicated_2021_2025.csv")

features = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos"
]

print("--- LOADING HISTORICAL TRAINING DATA ---")

df = pd.read_csv(csv_path)

# Generate seasonal features from acquisition dates
if "date" not in df.columns:
    raise ValueError("Historical CSV does not contain a date column.")

dates = pd.to_datetime(df["date"], errors="coerce")
doy = dates.dt.dayofyear

df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

print("Historical rows:", len(df))
print("Date conversion failures:", dates.isna().sum())

with rasterio.open(raster_path) as src:

    arrays = src.read()
    predictors = arrays[:7]
    observed_lst = arrays[7]

    valid = (
        np.all(np.isfinite(predictors), axis=0)
        & np.isfinite(observed_lst)
        & (arrays[4] == 2026)
    )

    print("\n2026 valid pixels:", np.count_nonzero(valid))

    print("\n--- FEATURE RANGE COMPARISON ---")

    for i, feature in enumerate(features):

        train = pd.to_numeric(
            df[feature], errors="coerce"
        ).dropna().to_numpy()

        current = predictors[i][valid]

        print(f"\n{feature}")

        print(
            "Historical min / p1 / median / p99 / max:",
            np.round(
                np.percentile(train, [0, 1, 50, 99, 100]), 5
            )
        )

        print(
            "2026 min / p1 / median / p99 / max:",
            np.round(
                np.percentile(current, [0, 1, 50, 99, 100]), 5
            )
        )

        outside = np.count_nonzero(
            (current < train.min()) |
            (current > train.max())
        )

        print("2026 pixels outside historical min-max:", outside)

    ndvi = predictors[0][valid]

    print("\n--- NDVI QUALITY CHECK ---")
    print("NDVI below -1:", np.count_nonzero(ndvi < -1))
    print("NDVI above +1:", np.count_nonzero(ndvi > 1))

print("\nComparison completed.")