import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

csv_path = Path("Bengaluru_UHI_Deduplicated_2021_2025.csv")

tif_path = Path(
    "Bengaluru_2026_Prediction/"
    "Bengaluru_2026_Cropland_Predictors_30m.tif"
)

features = [
    "LST_C",
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m"
]

df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["doy"] = df["date"].dt.day_of_year

# Compare historical observations within 30 days
# of the 3 November acquisition.
target_doy = 307

distance = np.abs(
    (df["doy"] - target_doy + 182.625) % 365.25 - 182.625
)

seasonal = df[distance <= 30].copy()

print("--- HISTORICAL DATA COVERAGE ---")
print("All historical records:", len(df))
print("Records within 30 days of 3 November:", len(seasonal))

if "product_id" in df.columns:
    print("Historical acquisitions in window:",
          seasonal["product_id"].nunique())

with rasterio.open(tif_path) as src:
    bands = src.read()

    valid = (
        np.all(np.isfinite(bands[:7]), axis=0)
        & np.isfinite(bands[7])
        & (bands[0] >= -1)
        & (bands[0] <= 1)
    )

    current = {
        "LST_C": bands[7][valid],
        "NDVI": bands[0][valid],
        "NDMI": bands[1][valid],
        "builtup_fraction_500m_2021": bands[2][valid],
        "elevation_m": bands[3][valid]
    }

print("\n--- FEATURE DISTRIBUTION COMPARISON ---")

for feature in features:

    historical_all = pd.to_numeric(
        df[feature], errors="coerce"
    ).dropna()

    historical_season = pd.to_numeric(
        seasonal[feature], errors="coerce"
    ).dropna()

    current_values = current[feature]

    print(f"\n{feature}")

    print("Historical all-years:",
          np.round(np.percentile(
              historical_all, [10, 25, 50, 75, 90]
          ), 3))

    if len(historical_season):
        print("Historical late-Oct/Nov/early-Dec:",
              np.round(np.percentile(
                  historical_season, [10, 25, 50, 75, 90]
              ), 3))

    print("2026 November scene:",
          np.round(np.percentile(
              current_values, [10, 25, 50, 75, 90]
          ), 3))

print("\nComparison completed.")