
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

import folium
from folium.plugins import Draw
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap

import pandas as pd
import numpy as np
import joblib
import plotly.express as px

try:
    import rasterio
    from rasterio.transform import array_bounds, xy
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.warp import transform as transform_coords
    from rasterio.warp import transform_geom
    RASTERIO_AVAILABLE = True
    RASTERIO_ERROR = None
except Exception as exc:
    RASTERIO_AVAILABLE = False
    RASTERIO_ERROR = str(exc)


# ======================================================
# 1. CONFIGURATION
# ======================================================

ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "Bengaluru_UHI_Best_Baseline_Model.joblib"
DATA_FILE = ROOT / "Bengaluru_UHI_Deduplicated_2021_2025.csv"

PREDICTION_FOLDER = ROOT / "Bengaluru_2026_Prediction"
RASTER_FILE = PREDICTION_FOLDER / "Bengaluru_2026_Cropland_Predictors_30m.tif"
PREDICTED_RASTER_FILE = PREDICTION_FOLDER / "Bengaluru_2026_Predicted_LST_30m.tif"
RESIDUAL_RASTER_FILE = PREDICTION_FOLDER / "Bengaluru_2026_Residuals_30m.tif"

FEATURES = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos",
]

BAND_NAMES = FEATURES + ["observed_LST_C"]

CENTER_LAT = 12.9716
CENTER_LON = 77.5946
NODATA_VALUE = -9999.0

st.set_page_config(
    page_title="Bengaluru Cropland LST Dashboard",
    page_icon="🌍",
    layout="wide",
)


# ======================================================
# 2. SHARED LOADERS
# ======================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_historical_data():
    df = pd.read_csv(DATA_FILE)

    required = [
        "date",
        "year",
        "product_id",
        "longitude",
        "latitude",
        "LST_C",
    ] + [f for f in FEATURES if f not in ["doy_sin", "doy_cos"]]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing historical dataset columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    day = df["date"].dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day / 365.25)

    numeric_columns = FEATURES + ["longitude", "latitude", "LST_C"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=numeric_columns + ["product_id", "date"]
    ).copy()

    return df


@st.cache_data(show_spinner="Preparing the 2026 raster map...")
def load_2026_map_preview(map_layer):
    """Read 2026 rasters and prepare a lightweight observed/predicted/residual overlay."""
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(f"Rasterio is unavailable: {RASTERIO_ERROR}")

    for path in [RASTER_FILE, PREDICTED_RASTER_FILE, RESIDUAL_RASTER_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required 2026 raster file not found: {path}")

    with rasterio.open(RASTER_FILE) as src:
        if src.count != 8:
            raise ValueError(f"Expected 8 predictor bands, found {src.count}.")
        if tuple(src.descriptions) != tuple(BAND_NAMES):
            raise ValueError(
                "The predictor raster band order does not match the model. "
                f"Found: {src.descriptions}"
            )

        height, width = src.height, src.width
        raster_crs = src.crs
        raster_transform = src.transform
        ndvi = src.read(1)
        year = src.read(5)
        observed = src.read(8)

        finite_predictors = (
            np.isfinite(ndvi)
            & (ndvi >= -1)
            & (ndvi <= 1)
            & np.isfinite(year)
            & (year == 2026)
        )

        for band_index in [2, 3, 4, 6, 7]:
            band_values = src.read(band_index)
            finite_predictors &= np.isfinite(band_values)
            del band_values

    with rasterio.open(PREDICTED_RASTER_FILE) as src:
        predicted = src.read(1)
        predicted_nodata = src.nodata

    with rasterio.open(RESIDUAL_RASTER_FILE) as src:
        residual = src.read(1)
        residual_nodata = src.nodata

    valid = (
        finite_predictors
        & np.isfinite(observed)
        & np.isfinite(predicted)
        & np.isfinite(residual)
        & (observed > 0)
        & (predicted != NODATA_VALUE)
        & (residual != NODATA_VALUE)
    )

    if predicted_nodata is not None:
        valid &= predicted != predicted_nodata
    if residual_nodata is not None:
        valid &= residual != residual_nodata

    if not np.any(valid):
        raise ValueError("No valid pixels were found in the 2026 prediction raster.")

    observed_values = observed[valid].astype(np.float64)
    predicted_values = predicted[valid].astype(np.float64)
    residual_values = residual[valid].astype(np.float64)

    ss_total = np.sum((observed_values - observed_values.mean()) ** 2)
    ss_res = np.sum(residual_values ** 2)
    r2 = 1 - ss_res / ss_total if ss_total > 0 else np.nan

    summary = {
        "valid_pixels": int(np.count_nonzero(valid)),
        "observed_mean": float(np.mean(observed_values)),
        "predicted_mean": float(np.mean(predicted_values)),
        "predicted_min": float(np.min(predicted_values)),
        "predicted_max": float(np.max(predicted_values)),
        "predicted_std": float(np.std(predicted_values)),
        "mae": float(np.mean(np.abs(residual_values))),
        "rmse": float(np.sqrt(np.mean(residual_values ** 2))),
        "r2": float(r2),
        "mean_residual": float(np.mean(residual_values)),
        "crs": str(raster_crs),
        "width": int(width),
        "height": int(height),
    }

    # Downsample only the displayed image.
    # Polygon statistics use the full-resolution raster.
    max_dimension = 760
    row_count = min(height, max_dimension)
    col_count = min(width, max_dimension)

    row_indices = np.linspace(0, height - 1, row_count).astype(int)
    col_indices = np.linspace(0, width - 1, col_count).astype(int)
    row_grid = np.ix_(row_indices, col_indices)

    if map_layer == "Observed LST":
        full_display_values = observed
        legend_caption = "Observed LST (°C)"
    elif map_layer == "Residual (observed − predicted)":
        full_display_values = residual
        legend_caption = "Residual: observed − predicted (°C)"
    else:
        full_display_values = predicted
        legend_caption = "Predicted LST (°C)"

    preview_values = full_display_values[row_grid].astype(np.float32)
    preview_valid = valid[row_grid] & np.isfinite(preview_values)
    display_values = preview_values[preview_valid]

    if map_layer == "Residual (observed − predicted)":
        limit = float(np.percentile(np.abs(display_values), 98))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0

        low, high = -limit, limit

        palette_hex = [
            "#2166ac", "#92c5de", "#f7f7f7",
            "#f4a582", "#b2182b"
        ]

        palette = np.array([
            [33, 102, 172],
            [146, 197, 222],
            [247, 247, 247],
            [244, 165, 130],
            [178, 24, 43],
        ], dtype=np.float32)

    else:
        low, high = np.percentile(display_values, [2, 98])

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(display_values))
            high = float(np.nanmax(display_values) + 0.01)

        palette_hex = [
            "#000004", "#420a68", "#932667",
            "#dd513a", "#fca50a", "#fcffa4"
        ]

        palette = np.array([
            [0, 0, 4],
            [66, 10, 104],
            [147, 38, 103],
            [221, 81, 58],
            [252, 165, 10],
            [252, 255, 164],
        ], dtype=np.float32)

    safe_values = np.where(preview_valid, preview_values, low)
    scaled = np.clip((safe_values - low) / (high - low), 0, 1)

    position = scaled * (len(palette) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(palette) - 1)
    fraction = (position - lower)[..., None]

    rgb = (
        palette[lower] * (1 - fraction)
        + palette[upper] * fraction
    ).astype(np.uint8)

    rgba = np.dstack([
        rgb,
        np.where(preview_valid, 205, 0).astype(np.uint8)
    ])

    left, bottom, right, top = array_bounds(
        height, width, raster_transform
    )

    corner_x = [left, right, right, left]
    corner_y = [bottom, bottom, top, top]

    corner_lon, corner_lat = transform_coords(
        raster_crs,
        "EPSG:4326",
        corner_x,
        corner_y,
    )

    image_bounds = [
        [float(min(corner_lat)), float(min(corner_lon))],
        [float(max(corner_lat)), float(max(corner_lon))],
    ]

    return (
        rgba,
        image_bounds,
        summary,
        float(low),
        float(high),
        palette_hex,
        legend_caption,
    )


# ======================================================
# 3. MAP AND SELECTION HELPERS
# ======================================================

def add_drawing_tools(map_object):
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(map_object)


def add_selected_geometry(map_object):
    selected = st.session_state.get("selected_polygon")

    if selected:
        folium.GeoJson(
            selected,
            name="Selected area",
            style_function=lambda feature: {
                "color": "#00A6A6",
                "weight": 3,
                "fillColor": "#00A6A6",
                "fillOpacity": 0.12,
            },
        ).add_to(map_object)


def render_map(map_object, mode_key):
    output = st_folium(
        map_object,
        height=620,
        width=None,
        key=f"bengaluru_map_{mode_key}_{st.session_state.map_version}",
        returned_objects=["last_active_drawing"],
    )

    drawing = output.get("last_active_drawing") if output else None

    if drawing and drawing.get("geometry"):
        if drawing != st.session_state.get("selected_polygon"):

            for state_key in list(st.session_state.keys()):
                if state_key.startswith("shap_result_") or state_key.startswith("shap_error_"):
                    del st.session_state[state_key]

            st.session_state.selected_polygon = drawing
            st.rerun()


def point_in_ring(longitude, latitude, ring):
    """Vectorized ray-casting test for a ring in GeoJSON lon/lat coordinates."""
    points = np.asarray(ring, dtype=float)

    x = points[:, 0]
    y = points[:, 1]

    inside = np.zeros(len(longitude), dtype=bool)
    j = len(points) - 1

    for i in range(len(points)):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]

        intersects = (
            ((yi > latitude) != (yj > latitude))
            & (
                longitude
                < (xj - xi) * (latitude - yi) / (yj - yi + 1e-12) + xi
            )
        )

        inside ^= intersects
        j = i

    return inside


def point_polygon_mask(longitude, latitude, geometry):
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]

    result = np.zeros(len(longitude), dtype=bool)

    polygons = [coordinates] if geometry_type == "Polygon" else coordinates

    if geometry_type not in ["Polygon", "MultiPolygon"]:
        raise ValueError("Please draw a polygon or rectangle.")

    for polygon_rings in polygons:
        polygon_mask = point_in_ring(
            longitude, latitude, polygon_rings[0]
        )

        for hole in polygon_rings[1:]:
            polygon_mask &= ~point_in_ring(longitude, latitude, hole)

        result |= polygon_mask

    return result


def selected_raster_pixels(geometry):
    """Extract full-resolution 30 m pixels inside the drawn polygon."""

    if not RASTERIO_AVAILABLE:
        raise RuntimeError(f"Rasterio is unavailable: {RASTERIO_ERROR}")

    with rasterio.open(RASTER_FILE) as src:
        geom_in_raster_crs = transform_geom(
            "EPSG:4326",
            src.crs,
            geometry,
            precision=3,
        )

        window = geometry_window(src, [geom_in_raster_crs])
        window = window.round_offsets().round_lengths()

        band_data = src.read(
            indexes=list(range(1, 9)),
            window=window,
        )

        window_transform = src.window_transform(window)
        raster_crs = src.crs

        inside = geometry_mask(
            [geom_in_raster_crs],
            out_shape=(band_data.shape[1], band_data.shape[2]),
            transform=window_transform,
            invert=True,
            all_touched=False,
        )

    with rasterio.open(PREDICTED_RASTER_FILE) as src:
        predicted_window = src.read(1, window=window)
        prediction_nodata = src.nodata

    with rasterio.open(RESIDUAL_RASTER_FILE) as src:
        residual_window = src.read(1, window=window)
        residual_nodata = src.nodata

    predictors = band_data[:7]
    observed = band_data[7]

    predicted = predicted_window
    residual = residual_window

    valid = (
        inside
        & np.all(np.isfinite(predictors), axis=0)
        & np.isfinite(observed)
        & np.isfinite(predicted)
        & np.isfinite(residual)
        & (predictors[0] >= -1)
        & (predictors[0] <= 1)
        & (predictors[4] == 2026)
        & (observed > 0)
        & (predicted != NODATA_VALUE)
        & (residual != NODATA_VALUE)
    )

    if prediction_nodata is not None:
        valid &= predicted != prediction_nodata

    if residual_nodata is not None:
        valid &= residual != residual_nodata

    rows, cols = np.where(valid)

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "longitude",
            "latitude",
            "observed_LST_C",
            "predicted_LST_C",
            "residual_C",
            *FEATURES,
        ])

    x_coords, y_coords = xy(
        window_transform,
        rows,
        cols,
        offset="center",
    )

    longitude, latitude = transform_coords(
        raster_crs,
        "EPSG:4326",
        x_coords,
        y_coords,
    )

    records = {
        "longitude": np.asarray(longitude),
        "latitude": np.asarray(latitude),
        "observed_LST_C": observed[valid].astype(np.float32),
        "predicted_LST_C": predicted[valid].astype(np.float32),
        "residual_C": residual[valid].astype(np.float32),
    }

    for index, feature in enumerate(FEATURES):
        records[feature] = predictors[index][valid].astype(np.float32)

    return pd.DataFrame(records)


# ======================================================
# 4. CHARTS AND EXPLANATIONS
# ======================================================

def show_selected_area_statistics(
    data,
    observed_column,
    predicted_column,
    residual_column,
):
    st.markdown("### Selected-area temperature statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Valid pixels / samples", f"{len(data):,}")
    c2.metric(
        "Mean predicted LST",
        f"{data[predicted_column].mean():.2f} °C"
    )
    c3.metric(
        "Minimum predicted LST",
        f"{data[predicted_column].min():.2f} °C"
    )
    c4.metric(
        "Maximum predicted LST",
        f"{data[predicted_column].max():.2f} °C"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Mean observed LST",
        f"{data[observed_column].mean():.2f} °C"
    )
    c6.metric(
        "Observed LST SD",
        f"{data[observed_column].std():.2f} °C"
    )
    c7.metric(
        "Predicted LST SD",
        f"{data[predicted_column].std():.2f} °C"
    )
    c8.metric(
        "Mean absolute residual",
        f"{data[residual_column].abs().mean():.2f} °C"
    )

    left, right = st.columns(2)

    with left:
        hist = px.histogram(
            data,
            x=predicted_column,
            nbins=30,
            title="Selected-area predicted LST distribution",
            labels={predicted_column: "Predicted LST (°C)"},
        )

        st.plotly_chart(hist, use_container_width=True)

    with right:
        compare_data = data

        if len(compare_data) > 15000:
            compare_data = compare_data.sample(
                15000, random_state=42
            )

        scatter = px.scatter(
            compare_data,
            x=observed_column,
            y=predicted_column,
            opacity=0.45,
            title="Observed versus predicted LST",
            labels={
                observed_column: "Observed LST (°C)",
                predicted_column: "Predicted LST (°C)",
            },
        )

        low = float(min(
            data[observed_column].min(),
            data[predicted_column].min(),
        ))

        high = float(max(
            data[observed_column].max(),
            data[predicted_column].max(),
        ))

        scatter.add_shape(
            type="line",
            x0=low,
            y0=low,
            x1=high,
            y1=high,
            line={"dash": "dash", "color": "red"},
        )

        st.plotly_chart(scatter, use_container_width=True)


def calculate_shap_table(model, feature_data, max_rows=300):
    """Return selected-area mean absolute and mean signed SHAP values."""

    try:
        import shap
    except Exception as exc:
        return None, f"SHAP could not be imported: {exc}"

    sample = (
        feature_data[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if sample.empty:
        return None, "No complete feature rows are available for SHAP."

    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)

        if isinstance(values, list):
            values = values[0]

        values = np.asarray(values)

        if values.ndim == 3:
            values = values[:, :, 0]

        result = pd.DataFrame({
            "Feature": FEATURES,
            "Mean absolute SHAP (°C)": np.mean(np.abs(values), axis=0),
            "Mean signed SHAP (°C)": np.mean(values, axis=0),
        }).sort_values(
            "Mean absolute SHAP (°C)",
            ascending=False
        )

        return result, None

    except Exception as exc:
        return None, f"SHAP calculation failed: {exc}"


def render_explanation_panel(model, feature_data, key_suffix):
    st.markdown(
        "### Explainable AI — selected-area feature contributions"
    )

    st.caption(
        "SHAP values are calculated for a reproducible sample of up to "
        "300 selected pixels. Mean absolute SHAP indicates contribution "
        "magnitude; mean signed SHAP indicates the average direction."
    )

    if st.button(
        "Calculate selected-area SHAP explanation",
        key=f"shap_button_{key_suffix}",
    ):
        with st.spinner(
            "Calculating TreeSHAP values for the selected area..."
        ):
            shap_table, error = calculate_shap_table(
                model, feature_data
            )

        if shap_table is not None:
            st.session_state[f"shap_result_{key_suffix}"] = shap_table
            st.session_state[f"shap_error_{key_suffix}"] = None
        else:
            st.session_state[f"shap_result_{key_suffix}"] = None
            st.session_state[f"shap_error_{key_suffix}"] = error

    prior_result = st.session_state.get(
        f"shap_result_{key_suffix}"
    )

    prior_error = st.session_state.get(
        f"shap_error_{key_suffix}"
    )

    if prior_result is not None:
        chart = px.bar(
            prior_result.sort_values(
                "Mean absolute SHAP (°C)"
            ),
            x="Mean absolute SHAP (°C)",
            y="Feature",
            orientation="h",
            title="Selected-area SHAP importance",
        )

        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(
            prior_result,
            use_container_width=True,
            hide_index=True,
        )

    elif prior_error:
        st.warning(prior_error)

        if hasattr(model, "feature_importances_"):
            fallback = pd.DataFrame({
                "Feature": FEATURES,
                "Random Forest global importance": model.feature_importances_,
            }).sort_values(
                "Random Forest global importance",
                ascending=False,
            )

            st.info(
                "The chart below is the model's global tree feature "
                "importance, not SHAP and not specific to the selected polygon."
            )

            st.bar_chart(fallback.set_index("Feature"))

    else:
        st.info(
            "Select the button to calculate SHAP contributions for this area."
        )


# ======================================================
# 5. INITIALIZE MODEL AND PAGE
# ======================================================

st.title("🌍 Bengaluru Cropland LST Prediction Dashboard")

st.markdown(
    "**Explainable Machine Learning for Peri-Urban Cropland "
    "Land Surface Temperature Analysis**"
)

if not MODEL_FILE.exists():
    st.error(f"Trained model file not found: {MODEL_FILE.name}")
    st.stop()

try:
    with st.spinner("Loading the saved Random Forest model..."):
        model = load_model()

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

if list(getattr(model, "feature_names_in_", FEATURES)) != FEATURES:
    st.error(
        "The saved model feature names/order do not match the dashboard configuration."
    )
    st.stop()

if "selected_polygon" not in st.session_state:
    st.session_state.selected_polygon = None

if "map_version" not in st.session_state:
    st.session_state.map_version = 0

st.sidebar.header("Analysis Settings")

analysis_mode = st.sidebar.radio(
    "Choose analysis",
    [
        "2026 continuous raster",
        "Historical acquisitions (2021–2025)",
    ],
    index=0,
)

col_clear, col_note = st.columns([1, 5])

with col_clear:
    if st.button("Clear map selection"):
        st.session_state.selected_polygon = None
        st.session_state.map_version += 1

        for state_key in list(st.session_state.keys()):
            if (
                state_key.startswith("shap_result_")
                or state_key.startswith("shap_error_")
            ):
                del st.session_state[state_key]

        st.rerun()


# ======================================================
# 6. 2026 CONTINUOUS RASTER MODE
# ======================================================

if analysis_mode == "2026 continuous raster":

    st.subheader(
        "2026 Landsat 9 — Continuous 30 m Cropland LST"
    )

    required_files = [
        RASTER_FILE,
        PREDICTED_RASTER_FILE,
        RESIDUAL_RASTER_FILE,
    ]

    missing_files = [
        path.name for path in required_files
        if not path.exists()
    ]

    if missing_files:
        st.error(
            "The following 2026 files are missing from "
            f"{PREDICTION_FOLDER.name}: {', '.join(missing_files)}"
        )
        st.stop()

    if not RASTERIO_AVAILABLE:
        st.error(
            "Rasterio is required for 2026 raster mode. "
            f"Details: {RASTERIO_ERROR}"
        )
        st.stop()

    try:
        selected_map_layer = st.sidebar.selectbox(
            "2026 map layer",
            [
                "Predicted LST",
                "Observed LST",
                "Residual (observed − predicted)",
            ],
        )

        (
            rgba,
            image_bounds,
            summary,
            map_vmin,
            map_vmax,
            legend_colours,
            legend_caption,
        ) = load_2026_map_preview(selected_map_layer)

    except Exception as error:
        st.error(f"Could not load 2026 raster layers: {error}")
        st.stop()

    st.caption(
        "Scene date: 3 November 2026 | Landsat 9 | "
        "30 m raster | Predictions use the saved 2021–2025 Random Forest model."
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Valid cropland pixels",
        f"{summary['valid_pixels']:,}"
    )

    k2.metric(
        "Mean observed LST",
        f"{summary['observed_mean']:.2f} °C"
    )

    k3.metric(
        "Mean predicted LST",
        f"{summary['predicted_mean']:.2f} °C"
    )

    k4.metric(
        "Mean absolute residual",
        f"{summary['mae']:.2f} °C"
    )

    st.warning(
        "Research caution: this is an exploratory transfer to one "
        "2026 scene, not a validated forecast. Whole-scene diagnostics: "
        f"RMSE {summary['rmse']:.2f} °C, MAE {summary['mae']:.2f} °C, "
        f"R² {summary['r2']:.3f}. The model tends to compress temperature extremes."
    )

    st.markdown(
        "#### Select an area from the 2026 prediction map"
    )

    st.write(
        f"The coloured overlay shows: {selected_map_layer}. "
        "Draw a polygon or rectangle; selected-area statistics will use "
        "the underlying full-resolution 30 m pixels, not the downsampled display image."
    )

    map_2026 = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    ImageOverlay(
        image=rgba,
        bounds=image_bounds,
        origin="upper",
        opacity=0.78,
        name=f"{selected_map_layer} — 2026",
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(map_2026)

    legend = LinearColormap(
        legend_colours,
        vmin=map_vmin,
        vmax=map_vmax,
    )

    legend.caption = (
        f"{legend_caption} — map display percentile stretch"
    )

    legend.add_to(map_2026)

    add_drawing_tools(map_2026)
    add_selected_geometry(map_2026)

    folium.LayerControl(collapsed=True).add_to(map_2026)

    render_map(map_2026, "2026")

    st.divider()
    st.subheader("Selected-area 2026 raster analysis")

    selected_geometry = st.session_state.get("selected_polygon")

    if not selected_geometry:
        st.info(
            "Draw a polygon or rectangle on the map to calculate "
            "statistics from the 30 m raster pixels."
        )

    else:
        try:
            with st.spinner(
                "Extracting all valid 30 m pixels inside the selected area..."
            ):
                selected_2026 = selected_raster_pixels(
                    selected_geometry["geometry"]
                )

        except Exception as error:
            selected_2026 = pd.DataFrame()
            st.error(
                f"Could not analyse the selected raster area: {error}"
            )

        if selected_2026.empty:
            st.warning(
                "No valid cropland pixels were found inside this polygon. "
                "Draw a larger area or move it over the valid cropland raster."
            )

        else:
            show_selected_area_statistics(
                selected_2026,
                "observed_LST_C",
                "predicted_LST_C",
                "residual_C",
            )

            st.markdown("### Selected 30 m pixel records")

            st.caption(
                f"The table previews up to 1,000 of {len(selected_2026):,} "
                "selected pixels. The CSV download contains all selected valid pixels."
            )

            st.dataframe(
                selected_2026.head(1000),
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = selected_2026.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download all selected 2026 raster pixels (CSV)",
                data=csv_bytes,
                file_name="Bengaluru_2026_Selected_Area_30m_Pixels.csv",
                mime="text/csv",
                key="download_2026_selected",
            )

            render_explanation_panel(
                model,
                selected_2026[FEATURES],
                "2026",
            )

    with st.expander("2026 scene details and interpretation"):

        st.write(
            f"""
            - Acquisition: Landsat 9, 3 November 2026.
            - Raster CRS: {summary['crs']}.
            - Raster dimensions: {summary['width']:,} × {summary['height']:,}.
            - Native pixel size: 30 m.
            - Valid prediction pixels: {summary['valid_pixels']:,}.
            - Residual definition: observed LST minus predicted LST.
            - Mean residual: {summary['mean_residual']:.2f} °C.
            - Predictions use the saved historical Random Forest model;
              the model was not retrained on 2026 data.
            - Only valid cropland pixels exported by the Earth Engine
              processing workflow are analysed.
            - The displayed map image is downsampled for browser performance;
              polygon statistics and downloads use full-resolution 30 m pixels.
            - The 2026 scene-level metrics are a temporal-transfer diagnostic,
              not an independent validation across the full year.
            """
        )


# ======================================================
# 7. HISTORICAL ACQUISITION MODE
# ======================================================

else:

    if not DATA_FILE.exists():
        st.error(
            f"Historical cropland dataset not found: {DATA_FILE.name}"
        )
        st.stop()

    try:
        with st.spinner(
            "Loading historical cropland observations..."
        ):
            historical_df = load_historical_data()

    except Exception as error:
        st.error(
            f"Could not load the historical dataset: {error}"
        )
        st.stop()

    products = sorted(historical_df["product_id"].unique())

    product_date = (
        historical_df[["product_id", "date"]]
        .drop_duplicates("product_id")
        .set_index("product_id")["date"]
    )

    selected_product = st.sidebar.selectbox(
        "Select Landsat acquisition",
        products,
        format_func=lambda product: (
            f"{product_date[product].strftime('%Y-%m-%d')} | {product}"
        ),
    )

    max_map_points = st.sidebar.slider(
        "Maximum historical points shown on map",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500,
    )

    scene = historical_df[
        historical_df["product_id"] == selected_product
    ].copy()

    if scene.empty:
        st.warning("No records were found for this acquisition.")
        st.stop()

    try:
        scene["predicted_LST_C"] = model.predict(
            scene[FEATURES].astype(float)
        )

        scene["residual_C"] = (
            scene["LST_C"] - scene["predicted_LST_C"]
        )

    except Exception as error:
        st.error(f"Historical prediction failed: {error}")
        st.stop()

    st.subheader("Selected historical acquisition overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Acquisition date",
        scene["date"].iloc[0].strftime("%d %b %Y")
    )

    c2.metric(
        "Sampled locations",
        f"{len(scene):,}"
    )

    c3.metric(
        "Mean observed LST",
        f"{scene['LST_C'].mean():.2f} °C"
    )

    c4.metric(
        "Mean predicted LST",
        f"{scene['predicted_LST_C'].mean():.2f} °C"
    )

    st.info(
        "Historical mode displays sampled cropland pixel-date observations. "
        "It is not a continuous 30 m raster."
    )

    st.markdown("#### Select a sampled cropland area")

    historical_map = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    vmin = float(scene["predicted_LST_C"].min())
    vmax = float(scene["predicted_LST_C"].max())

    if vmax <= vmin:
        vmax = vmin + 0.01

    colour_scale = LinearColormap(
        ["#2166ac", "#67a9cf", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=vmin,
        vmax=vmax,
    )

    colour_scale.caption = "Predicted LST (°C)"
    colour_scale.add_to(historical_map)

    map_data = (
        scene if len(scene) <= max_map_points
        else scene.sample(max_map_points, random_state=42)
    )

    for _, row in map_data.iterrows():

        predicted_value = float(row["predicted_LST_C"])

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=3,
            color=colour_scale(predicted_value),
            fill=True,
            fill_color=colour_scale(predicted_value),
            fill_opacity=0.8,
            weight=1,
            popup=(
                f"Observed LST: {row['LST_C']:.2f} °C<br>"
                f"Predicted LST: {predicted_value:.2f} °C<br>"
                f"Date: {row['date'].strftime('%Y-%m-%d')}"
            ),
        ).add_to(historical_map)

    add_drawing_tools(historical_map)
    add_selected_geometry(historical_map)

    render_map(historical_map, "historical")

    st.divider()
    st.subheader("Selected-area historical analysis")

    selected_geometry = st.session_state.get("selected_polygon")

    if not selected_geometry:
        st.info(
            "Draw a polygon or rectangle to analyse the sampled locations inside it."
        )

    else:
        try:
            geometry = selected_geometry["geometry"]

            mask = point_polygon_mask(
                scene["longitude"].to_numpy(),
                scene["latitude"].to_numpy(),
                geometry,
            )

            selected_historical = scene.loc[mask].copy()

        except Exception as error:
            st.error(f"Area selection error: {error}")
            selected_historical = pd.DataFrame()

        if selected_historical.empty:
            st.warning(
                "No sampled cropland locations were found inside this polygon "
                "for the selected acquisition. Try a larger area or another date."
            )

        else:
            show_selected_area_statistics(
                selected_historical,
                "LST_C",
                "predicted_LST_C",
                "residual_C",
            )

            st.markdown("### Selected historical sample records")

            display_columns = [
                "date",
                "longitude",
                "latitude",
                "LST_C",
                "predicted_LST_C",
                "residual_C",
                *FEATURES,
            ]

            st.dataframe(
                selected_historical[display_columns].head(1000),
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = selected_historical[
                display_columns
            ].to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download selected historical samples (CSV)",
                data=csv_bytes,
                file_name="Bengaluru_Selected_Historical_Area_LST.csv",
                mime="text/csv",
                key="download_historical_selected",
            )

            render_explanation_panel(
                model,
                selected_historical[FEATURES],
                "historical",
            )

    with st.expander("Historical mode interpretation"):

        st.write(
            "Historical predictions use the saved Random Forest model "
            "and the selected acquisition's sampled cropland records. "
            "Polygon statistics refer only to the sampled records inside "
            "the boundary. The selected acquisition is not automatically "
            "an independent test set; consult the project's spatial, "
            "scene-held-out and temporal validation results when reporting "
            "model performance."
        )


# ======================================================
# 8. SHARED FOOTER
# ======================================================

st.divider()

st.caption(
    "Bengaluru Cropland LST Prediction | Landsat 8/9 | "
    "Historical analysis 2021–2025 | 2026 exploratory raster transfer | "
    "Random Forest + Explainable AI"
)
