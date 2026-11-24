import numpy as np
import pandas as pd
import rasterio

from pathlib import Path
from rasterio.warp import transform
from rasterio.transform import xy
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------
# PATHS
# -------------------------------------------------

folder = Path("Bengaluru_2026_Prediction")

csv_path = Path("Bengaluru_UHI_Deduplicated_2021_2025.csv")
input_path = folder / "Bengaluru_2026_Cropland_Predictors_30m.tif"
prediction_path = folder / "Bengaluru_2026_Predicted_LST_30m.tif"

output_path = folder / "Bengaluru_2026_Model_vs_Nov_Climatology.csv"

TARGET_DOY = 307
SEASON_WINDOW = 30
MAX_DISTANCE_M = 45

# -------------------------------------------------
# LOAD HISTORICAL DATA
# -------------------------------------------------

print("Loading historical data...")

df = pd.read_csv(csv_path)

required = ["date", "longitude", "latitude", "LST_C"]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["longitude", "latitude", "LST_C"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["doy"] = df["date"].dt.day_of_year

distance = np.abs(
    (df["doy"] - TARGET_DOY + 182.625) % 365.25 - 182.625
)

season = df.loc[distance <= SEASON_WINDOW].copy()

season = season.dropna(
    subset=["longitude", "latitude", "LST_C"]
)

season = season[
    np.isfinite(season["LST_C"])
]

print("Historical seasonal records:", len(season))

if len(season) == 0:
    raise ValueError("No historical seasonal records found.")

# -------------------------------------------------
# READ 2026 RASTER AND PREDICTIONS
# -------------------------------------------------

print("\nReading 2026 raster...")

with rasterio.open(input_path) as src:

    bands = src.read()
    raster_crs = src.crs
    raster_transform = src.transform

    predictors = bands[:7]
    observed_array = bands[7]

    valid = (
        np.all(np.isfinite(predictors), axis=0)
        & np.isfinite(observed_array)
        & (predictors[0] >= -1)
        & (predictors[0] <= 1)
        & (predictors[4] == 2026)
        & (observed_array > 0)
    )

    rows, cols = np.where(valid)

    observed = observed_array[valid]

    target_x, target_y = xy(
        raster_transform,
        rows,
        cols,
        offset="center"
    )

    target_xy = np.column_stack([
        target_x, target_y
    ])

with rasterio.open(prediction_path) as src:
    predicted_array = src.read(1)

predicted = predicted_array[valid]

prediction_valid = (
    np.isfinite(predicted)
    & (predicted != -9999)
)

target_xy = target_xy[prediction_valid]
observed = observed[prediction_valid]
predicted = predicted[prediction_valid]

print("Valid 2026 pixels:", len(observed))

# -------------------------------------------------
# PROJECT HISTORICAL COORDINATES
# -------------------------------------------------

print("\nProjecting historical coordinates...")

hx, hy = transform(
    "EPSG:4326",
    raster_crs,
    season["longitude"].to_list(),
    season["latitude"].to_list()
)

history = pd.DataFrame({
    "x": np.asarray(hx),
    "y": np.asarray(hy),
    "LST_C": season["LST_C"].to_numpy()
})

history = history.replace(
    [np.inf, -np.inf], np.nan
).dropna()

# Aggregate repeated observations at the same
# approximate location into a seasonal median.
history["x_m"] = np.rint(history["x"]).astype(int)
history["y_m"] = np.rint(history["y"]).astype(int)

climatology = (
    history.groupby(["x_m", "y_m"])["LST_C"]
    .median()
    .reset_index()
)

history_xy = climatology[
    ["x_m", "y_m"]
].to_numpy(dtype=float)

history_lst = climatology[
    "LST_C"
].to_numpy(dtype=float)

print("Unique historical seasonal locations:", len(history_lst))

if len(history_lst) == 0:
    raise ValueError("No usable historical locations.")

# -------------------------------------------------
# FIND NEAREST HISTORICAL LOCATION
# -------------------------------------------------

print("\nMatching 2026 pixels with historical locations...")

tree = cKDTree(history_xy)

distances, indices = tree.query(
    target_xy,
    k=1,
    distance_upper_bound=MAX_DISTANCE_M
)

matched = (
    np.isfinite(distances)
    & (indices < len(history_lst))
)

print("Matched pixels:", np.count_nonzero(matched))
print("Unmatched pixels:", np.count_nonzero(~matched))
print(
    "Match coverage:",
    round(100 * np.mean(matched), 2),
    "%"
)

if np.count_nonzero(matched) < 100:
    raise ValueError(
        "Too few spatial matches. Do not interpret the comparison."
    )

# -------------------------------------------------
# EVALUATE BOTH METHODS ON SAME PIXELS
# -------------------------------------------------

obs = observed[matched]
rf = predicted[matched]
clim = history_lst[indices[matched]]

def evaluate(name, actual, estimate):

    rmse = np.sqrt(
        mean_squared_error(actual, estimate)
    )

    mae = mean_absolute_error(actual, estimate)

    r2 = r2_score(actual, estimate)

    bias = np.mean(estimate - actual)

    return {
        "Method": name,
        "Pixel_Count": len(actual),
        "RMSE_C": rmse,
        "MAE_C": mae,
        "R2": r2,
        "Prediction_Bias_C": bias
    }

results = pd.DataFrame([
    evaluate("Random Forest 2026", obs, rf),
    evaluate("Historical November Climatology", obs, clim)
])

print("\n--- MODEL VS CLIMATOLOGY ---")
print(results.round(4).to_string(index=False))

print("\nMedian matching distance (m):",
      round(float(np.median(distances[matched])), 2))

print("95th percentile matching distance (m):",
      round(float(np.percentile(distances[matched], 95)), 2))

results.to_csv(output_path, index=False)

print("\nResults saved:", output_path)
print("\nComparison completed.")