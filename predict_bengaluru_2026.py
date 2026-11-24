import numpy as np
import pandas as pd
import rasterio
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

folder = Path("Bengaluru_2026_Prediction")

raster_path = folder / "Bengaluru_2026_Cropland_Predictors_30m.tif"
model_path = Path("Bengaluru_UHI_Best_Baseline_Model.joblib")

prediction_path = folder / "Bengaluru_2026_Predicted_LST_30m.tif"
residual_path = folder / "Bengaluru_2026_Residuals_30m.tif"
summary_path = folder / "Bengaluru_2026_Prediction_Summary.csv"

FEATURES = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos"
]

NODATA = -9999.0

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading saved Random Forest model...")

model = joblib.load(model_path)

if list(model.feature_names_in_) != FEATURES:
    raise ValueError("Model feature names/order do not match!")

print("Model loaded:", type(model).__name__)

# --------------------------------------------------
# READ RASTER
# --------------------------------------------------

print("\nReading 2026 predictor raster...")

with rasterio.open(raster_path) as src:

    bands = src.read()
    profile = src.profile.copy()

    height = src.height
    width = src.width

    if src.count != 8:
        raise ValueError("Expected 8 raster bands.")

    if list(src.descriptions) != FEATURES + ["observed_LST_C"]:
        raise ValueError("Raster band names/order do not match!")

    predictors = bands[:7]
    observed_lst = bands[7]

    # All seven predictors and observed LST must be finite
    finite_mask = (
        np.all(np.isfinite(predictors), axis=0)
        & np.isfinite(observed_lst)
        & (predictors[4] == 2026)
        & (observed_lst > 0)
    )

    # NDVI must be physically valid
    ndvi_mask = (
        (predictors[0] >= -1)
        & (predictors[0] <= 1)
    )

    valid_mask = finite_mask & ndvi_mask

    total_pixels = height * width
    finite_pixels = int(np.count_nonzero(finite_mask))
    invalid_ndvi = int(np.count_nonzero(finite_mask & ~ndvi_mask))
    valid_pixels = int(np.count_nonzero(valid_mask))

    print("Total raster pixels:", total_pixels)
    print("Finite candidate pixels:", finite_pixels)
    print("Excluded invalid NDVI pixels:", invalid_ndvi)
    print("Pixels for prediction:", valid_pixels)

    if valid_pixels == 0:
        raise ValueError("No valid pixels available for prediction.")

    # Prepare predictor matrix
    X = np.column_stack([
        predictors[i][valid_mask]
        for i in range(7)
    ])

    observed = observed_lst[valid_mask].astype(np.float32)

    # --------------------------------------------------
    # GENERATE PREDICTIONS IN BATCHES
    # --------------------------------------------------

    print("\nGenerating 2026 predictions...")

    predictions = np.empty(valid_pixels, dtype=np.float32)

    batch_size = 25000

    for start in range(0, valid_pixels, batch_size):

        end = min(start + batch_size, valid_pixels)

        batch = pd.DataFrame(
            X[start:end],
            columns=FEATURES
        )

        predictions[start:end] = model.predict(batch)

        print(f"Processed {end:,} / {valid_pixels:,} pixels")

    # --------------------------------------------------
    # CREATE OUTPUT RASTERS
    # --------------------------------------------------

    prediction_array = np.full(
        (height, width),
        NODATA,
        dtype=np.float32
    )

    residual_array = np.full(
        (height, width),
        NODATA,
        dtype=np.float32
    )

    prediction_array[valid_mask] = predictions

    # Residual = observed LST - predicted LST
    residuals = observed - predictions
    residual_array[valid_mask] = residuals

    output_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": src.crs,
        "transform": src.transform,
        "nodata": NODATA,
        "compress": "deflate"
    }

    with rasterio.open(
        prediction_path, "w", **output_profile
    ) as dst:
        dst.write(prediction_array, 1)
        dst.set_band_description(1, "Predicted_LST_C")

    with rasterio.open(
        residual_path, "w", **output_profile
    ) as dst:
        dst.write(residual_array, 1)
        dst.set_band_description(1, "Observed_minus_Predicted_C")

# --------------------------------------------------
# EVALUATE AGAINST OBSERVED 2026 LST
# --------------------------------------------------

rmse = float(np.sqrt(mean_squared_error(observed, predictions)))
mae = float(mean_absolute_error(observed, predictions))
r2 = float(r2_score(observed, predictions))
mean_error = float(np.mean(residuals))

print("\n--- 2026 PREDICTION RESULTS ---")

print("Valid predicted pixels:", valid_pixels)
print("Observed mean LST:", round(float(np.mean(observed)), 3))
print("Predicted mean LST:", round(float(np.mean(predictions)), 3))
print("Predicted minimum LST:", round(float(np.min(predictions)), 3))
print("Predicted maximum LST:", round(float(np.max(predictions)), 3))
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))
print("R2:", round(r2, 4))
print("Mean residual (observed - predicted):", round(mean_error, 4))

# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

summary = pd.DataFrame([{
    "prediction_year": 2026,
    "acquisition_date": "2026-11-03",
    "satellite": "Landsat 9",
    "valid_pixels": valid_pixels,
    "excluded_invalid_ndvi_pixels": invalid_ndvi,
    "observed_mean_LST_C": float(np.mean(observed)),
    "predicted_mean_LST_C": float(np.mean(predictions)),
    "predicted_min_LST_C": float(np.min(predictions)),
    "predicted_max_LST_C": float(np.max(predictions)),
    "RMSE_C": rmse,
    "MAE_C": mae,
    "R2": r2,
    "mean_residual_C": mean_error
}])

summary.to_csv(summary_path, index=False)

print("\n--- FILES SAVED ---")
print(prediction_path)
print(residual_path)
print(summary_path)

print("\n2026 prediction completed successfully.")