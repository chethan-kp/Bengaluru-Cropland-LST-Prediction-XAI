
# Bengaluru Cropland LST Prediction using Explainable AI

An explainable geospatial machine learning framework for predicting Land Surface Temperature (LST) over peri-urban cropland in Bengaluru, Karnataka, India.

The project combines Landsat 8/9 satellite imagery, Random Forest regression, SHAP explainability, Surface Urban Heat Island (SUHI) analysis and an interactive Streamlit dashboard.

## Project Overview

The project has two primary analytical components:

1. Cropland LST prediction using historical satellite observations from 2021–2025.
2. Daytime urban–rural surface thermal contrast analysis.

An additional exploratory experiment applies the trained historical model to a Landsat 9 acquisition from November 2026.

## Technologies

- Python
- Google Earth Engine
- Landsat 8 and Landsat 9
- Random Forest Regression
- Linear Regression
- Scikit-learn
- SHAP
- Rasterio
- Pandas and NumPy
- Streamlit
- Folium
- Plotly
- QGIS-compatible GeoTIFF outputs

## Dataset

Historical period: 2021–2025

- 284,745 pixel-date observations
- 73 accepted Landsat acquisitions
- Study area: Bengaluru, Karnataka
- Modelling domain: 10–30 km from the city centre

Data sources:

- USGS Landsat Collection 2 Level-2
- ESA WorldCover 2021 v200
- SRTM elevation data

## Methodology

### 1. Satellite preprocessing

- Landsat image collection filtering
- Cloud and quality masking
- Surface reflectance conversion
- Land Surface Temperature calculation
- Cropland identification

### 2. Feature engineering

Seven input features:

1. NDVI
2. NDMI
3. Built-up fraction (500 m)
4. Elevation
5. Year
6. Day-of-year sine
7. Day-of-year cosine

### 3. Machine learning

Algorithms:

- Linear Regression
- Random Forest Regression

Selected Random Forest configuration:

- 150 trees
- Maximum depth: 18
- Minimum samples per leaf: 5
- Max features: 1.0
- Random state: 42

### 4. Model evaluation

Spatial block, scene-held-out and temporal validation were performed using regression metrics.

### 5. Explainable AI

SHAP was used to understand feature contributions to Random Forest predictions.

Seasonal predictors and NDMI showed the largest global mean absolute SHAP contributions.

### 6. SUHI analysis

Daytime surface-temperature contrast was measured between:

- Urban built-up areas within 10 km.
- Cropland between 10 and 30 km.

A total of 115 valid acquisitions were analysed.

## Results

### Historical spatial test

| Metric | Result |
|---|---:|
| R² | 0.8822 |
| MAE | 1.4304 °C |
| RMSE | 1.8743 °C |

Note: Spatial validation has temporal overlap across spatial blocks. These values should not be interpreted as universal or future-year accuracy.

### SUHI analysis

| Metric | Result |
|---|---:|
| Valid acquisitions | 115 |
| Mean urban-rural contrast | -1.3414 °C |
| Median contrast | -1.5841 °C |

Negative contrast means the selected central built-up surfaces were cooler than the selected outer cropland surfaces in the sampled daytime scenes. This is not an atmospheric air-temperature UHI measurement.

### 2026 exploratory extension

Landsat 9 acquisition: 3 November 2026.

| Metric | Result |
|---|---:|
| Valid predicted pixels | 169,875 |
| R² | 0.0536 |
| MAE | 3.5004 °C |
| RMSE | 4.5378 °C |

This is an exploratory single-scene temporal-transfer experiment, not a validated annual forecast.

## Interactive Dashboard

The Streamlit application supports:

- Historical acquisition selection.
- Continuous 2026 LST raster visualization.
- Observed, predicted and residual map layers.
- Polygon and rectangle selection.
- Full-resolution selected-area statistics.
- Temperature distribution charts.
- SHAP explanations.
- Selected-area CSV downloads.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Bengaluru-Cropland-LST-Prediction-XAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Important Files

- `app.py` - Interactive dashboard.
- `Bengaluru_UHI_Best_Baseline_Model.joblib` - Trained Random Forest model.
- `Bengaluru_UHI_Deduplicated_2021_2025.csv` - Historical model dataset.
- `Bengaluru_2026_Prediction/` - 2026 raster predictions and diagnostics.
- `Bengaluru_SUHI_Analysis/` - SUHI results.
- `Bengaluru_UHI_Spatial_SHAP_Maps/` - Explainability visualizations.

## Limitations

- Spatial validation has temporal overlap.
- Temporal generalization is weaker than spatial test performance.
- The 2026 extension has weak predictive performance.
- The Random Forest compresses observed temperature extremes.
- SHAP explains model behaviour, not causation.
- Daytime surface temperature is not equivalent to air temperature.

## Author

Chethan K P  
Computer Science and Engineering (AI & ML)  
Dayananda Sagar University  
Bengaluru, India
