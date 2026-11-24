
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

DATA_FILE = FOLDER / "Bengaluru_UHI_Spatial_Split_2021_2025.csv"
MODEL_FILE = FOLDER / "Bengaluru_UHI_Best_Baseline_Model.joblib"

SAMPLE_SIZE = 500
RANDOM_STATE = 42

FEATURES = [
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "year",
    "doy_sin",
    "doy_cos",
]


# ==================================================
# 2. LOAD THE DATA AND TRAINED MODEL
# ==================================================

def main():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    data = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    print("FILES LOADED")
    print("-" * 50)
    print(f"Dataset: {DATA_FILE.name}")
    print(f"Model:   {MODEL_FILE.name}")

    required = FEATURES + ["LST_C", "split"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Use only the spatially held-out test data
    test = data[
        data["split"].astype(str).str.lower() == "test"
    ].copy()

    if test.empty:
        raise ValueError("No test rows found.")

    # Validate numeric data
    for column in FEATURES + ["LST_C"]:
        test[column] = pd.to_numeric(
            test[column], errors="coerce"
        )

    test = test.replace([np.inf, -np.inf], np.nan)
    test = test.dropna(subset=FEATURES + ["LST_C"])

    if test.empty:
        raise ValueError("No valid test rows remain.")

    # Confirm the model's feature order, if available
    saved_features = getattr(
        model, "feature_names_in_", None
    )

    if saved_features is not None:
        if list(saved_features) != FEATURES:
            raise ValueError(
                "The saved model feature order differs "
                "from the configured feature order."
            )

    print(f"Available test rows: {len(test):,}")

    # ==================================================
    # 3. SELECT A REPRODUCIBLE TEST SAMPLE
    # ==================================================

    n_samples = min(SAMPLE_SIZE, len(test))

    sample = test.sample(
        n=n_samples,
        random_state=RANDOM_STATE
    ).copy()

    X_sample = sample[FEATURES]

    print(f"Rows selected for SHAP: {len(sample):,}")

    # ==================================================
    # 4. CALCULATE SHAP VALUES
    # ==================================================

    print("\nCalculating SHAP values...")
    print("This explains the saved Random Forest model.")

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )

    explanation = explainer(X_sample)

    shap_array = np.asarray(explanation.values)

    # Handle a possible single-output third dimension
    if shap_array.ndim == 3 and shap_array.shape[-1] == 1:
        shap_array = shap_array[:, :, 0]

    if shap_array.shape != X_sample.shape:
        raise ValueError(
            f"Unexpected SHAP array shape: {shap_array.shape}"
        )

    # ==================================================
    # 5. GLOBAL SHAP FEATURE IMPORTANCE
    # ==================================================

    importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_SHAP_C": np.abs(
            shap_array
        ).mean(axis=0),
        "mean_SHAP_C": shap_array.mean(axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_SHAP_C",
        ascending=False
    ).reset_index(drop=True)

    importance_file = (
        FOLDER / "Bengaluru_UHI_SHAP_Global_Importance.csv"
    )

    importance.to_csv(importance_file, index=False)

    print("\nGLOBAL SHAP IMPORTANCE")
    print("-" * 70)
    print(importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    ))

    # ==================================================
    # 6. SAVE INDIVIDUAL TEST-SAMPLE EXPLANATIONS
    # ==================================================

    sample_output = sample.copy()

    sample_predictions = model.predict(X_sample)

    sample_output["LST_C_predicted"] = sample_predictions

    sample_output["prediction_residual_C"] = (
        sample_output["LST_C"] - sample_predictions
    )

    for i, feature in enumerate(FEATURES):
        sample_output[f"SHAP_{feature}_C"] = shap_array[:, i]

    base_values = np.asarray(
        explanation.base_values
    ).reshape(-1)

    if len(base_values) == 1:
        base_values = np.repeat(
            base_values[0], len(sample_output)
        )

    sample_output["SHAP_base_value_C"] = base_values

    sample_output["SHAP_reconstructed_prediction_C"] = (
        base_values + shap_array.sum(axis=1)
    )

    shap_output_file = (
        FOLDER / "Bengaluru_UHI_SHAP_Sample_Explanations.csv"
    )

    sample_output.to_csv(
        shap_output_file, index=False
    )

    # Verify SHAP additivity against model predictions
    max_difference = np.max(np.abs(
        sample_predictions
        - sample_output["SHAP_reconstructed_prediction_C"].to_numpy()
    ))

    print(
        "\nMaximum SHAP reconstruction difference: "
        f"{max_difference:.8f} °C"
    )

    # ==================================================
    # 7. CREATE SHAP BEESWARM SUMMARY PLOT
    # ==================================================

    beeswarm_file = (
        FOLDER / "Bengaluru_UHI_SHAP_Beeswarm.png"
    )

    shap.plots.beeswarm(
        explanation,
        max_display=len(FEATURES),
        show=False
    )

    plt.gcf().set_size_inches(11, 7)
    plt.tight_layout()
    plt.savefig(
        beeswarm_file,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close("all")

    # ==================================================
    # 8. CREATE SHAP GLOBAL IMPORTANCE BAR PLOT
    # ==================================================

    bar_file = (
        FOLDER / "Bengaluru_UHI_SHAP_Bar.png"
    )

    shap.plots.bar(
        explanation,
        max_display=len(FEATURES),
        show=False
    )

    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    plt.savefig(
        bar_file,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close("all")

    # ==================================================
    # 9. COMPLETION SUMMARY
    # ==================================================

    print("\nSHAP ANALYSIS COMPLETED")
    print("-" * 50)

    print(f"Global importance: {importance_file.name}")
    print(f"Sample explanations: {shap_output_file.name}")
    print(f"Beeswarm plot: {beeswarm_file.name}")
    print(f"Importance bar plot: {bar_file.name}")


if __name__ == "__main__":
    main()
