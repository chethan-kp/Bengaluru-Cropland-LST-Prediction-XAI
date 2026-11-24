import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

folder = Path("Bengaluru_2026_Prediction")

observed_file = folder / "Bengaluru_2026_Cropland_Predictors_30m.tif"
predicted_file = folder / "Bengaluru_2026_Predicted_LST_30m.tif"

with rasterio.open(observed_file) as src:
    observed = src.read(8)

with rasterio.open(predicted_file) as src:
    predicted = src.read(1)

valid = (
    np.isfinite(observed)
    & np.isfinite(predicted)
    & (predicted != -9999)
    & (observed > 0)
)

obs = observed[valid]
pred = predicted[valid]

residual = obs - pred
absolute_error = np.abs(residual)

bins = [-np.inf, 25, 30, 35, 40, np.inf]
labels = ["Below 25", "25–30", "30–35", "35–40", "Above 40"]

temperature_group = pd.cut(
    obs,
    bins=bins,
    labels=labels,
    right=False
)

df = pd.DataFrame({
    "Observed_Temperature_Range": temperature_group,
    "Observed_LST": obs,
    "Predicted_LST": pred,
    "Residual": residual,
    "Absolute_Error": absolute_error
})

results = df.groupby(
    "Observed_Temperature_Range",
    observed=False
).agg(
    Pixel_Count=("Observed_LST", "count"),
    Mean_Observed=("Observed_LST", "mean"),
    Mean_Predicted=("Predicted_LST", "mean"),
    Mean_Residual=("Residual", "mean"),
    MAE=("Absolute_Error", "mean")
).round(3)

output = folder / "Bengaluru_2026_Error_By_Temperature.csv"

results.to_csv(output)

print("\n--- ERROR BY OBSERVED TEMPERATURE ---")
print(results)

print("\nSaved:", output)