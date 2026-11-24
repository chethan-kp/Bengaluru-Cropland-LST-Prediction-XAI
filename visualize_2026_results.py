import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("Bengaluru_2026_Prediction")

input_file = folder / "Bengaluru_2026_Cropland_Predictors_30m.tif"
pred_file = folder / "Bengaluru_2026_Predicted_LST_30m.tif"
res_file = folder / "Bengaluru_2026_Residuals_30m.tif"

with rasterio.open(input_file) as src:
    observed = src.read(8).astype(float)
    transform = src.transform
    height, width = src.height, src.width

with rasterio.open(pred_file) as src:
    predicted = src.read(1).astype(float)

with rasterio.open(res_file) as src:
    residual = src.read(1).astype(float)

# Convert NoData and invalid values to NaN
observed[~np.isfinite(observed)] = np.nan
predicted[predicted == -9999] = np.nan
residual[residual == -9999] = np.nan

# Plot at reduced resolution for faster rendering
step = max(1, int(max(height, width) / 1000))

obs = observed[::step, ::step]
pred = predicted[::step, ::step]
res = residual[::step, ::step]

# Common display limits for observed and predicted LST
valid_values = np.concatenate([
    observed[np.isfinite(observed)],
    predicted[np.isfinite(predicted)]
])

vmin, vmax = np.percentile(valid_values, [2, 98])

res_limit = np.nanpercentile(np.abs(res), 98)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

im1 = axes[0, 0].imshow(
    obs, cmap="inferno", vmin=vmin, vmax=vmax
)
axes[0, 0].set_title("Observed LST - 3 Nov 2026")
plt.colorbar(im1, ax=axes[0, 0], label="Temperature (°C)")

im2 = axes[0, 1].imshow(
    pred, cmap="inferno", vmin=vmin, vmax=vmax
)
axes[0, 1].set_title("Random Forest Predicted LST")
plt.colorbar(im2, ax=axes[0, 1], label="Temperature (°C)")

im3 = axes[1, 0].imshow(
    res, cmap="coolwarm",
    vmin=-res_limit, vmax=res_limit
)
axes[1, 0].set_title("Residual: Observed - Predicted")
plt.colorbar(im3, ax=axes[1, 0], label="Error (°C)")

# Scatter plot
valid = np.isfinite(observed) & np.isfinite(predicted)

rng = np.random.default_rng(42)
indices = np.flatnonzero(valid)

if len(indices) > 10000:
    indices = rng.choice(indices, 10000, replace=False)

axes[1, 1].scatter(
    observed.ravel()[indices],
    predicted.ravel()[indices],
    s=3,
    alpha=0.3
)

axes[1, 1].plot([10, 50], [10, 50], "r--")
axes[1, 1].set_xlim(10, 50)
axes[1, 1].set_ylim(10, 50)
axes[1, 1].set_xlabel("Observed LST (°C)")
axes[1, 1].set_ylabel("Predicted LST (°C)")
axes[1, 1].set_title("Observed vs Predicted")

fig.suptitle(
    "Bengaluru 2026 Cropland LST Analysis",
    fontsize=16
)

plt.tight_layout()

output = folder / "Bengaluru_2026_Visual_Diagnostics.png"
plt.savefig(output, dpi=200, bbox_inches="tight")

print("Diagnostic map saved:", output)

plt.show()