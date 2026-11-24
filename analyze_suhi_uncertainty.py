from __future__ import annotations

from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_BASENAME = "Bengaluru_SUHI_Intensity_2021_2025"
OUTPUT_DIR = BASE_DIR / "Bengaluru_SUHI_Uncertainty_Results"
MIN_VALID_PIXELS = 1000
N_BOOTSTRAPS = 10000
RANDOM_SEED = 42
BLOCK_SIZES = (3, 5, 7)

REQUIRED = [
    "date",
    "product_id",
    "urban_mean_LST_C",
    "urban_valid_pixels",
    "rural_mean_LST_C",
    "rural_valid_pixels",
    "SUHI_intensity_C",
]


def locate_input() -> Path:
    candidates = [
        BASE_DIR / f"{INPUT_BASENAME}.csv",
        BASE_DIR / f"{INPUT_BASENAME}.xlsx",
        BASE_DIR / f"{INPUT_BASENAME}.xls",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Allow small naming differences while avoiding accidental use of unrelated files.
    matches = []
    for pattern in ("*SUHI*Intensity*2021*2025*.csv", "*SUHI*Intensity*2021*2025*.xlsx", "*SUHI*Intensity*2021*2025*.xls"):
        matches.extend(BASE_DIR.glob(pattern))
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(f" - {p.name}" for p in matches)
        raise FileNotFoundError(
            "More than one possible SUHI input file was found. Keep only the intended file beside this script,\n"
            "or rename it to Bengaluru_SUHI_Intensity_2021_2025.csv / .xlsx.\n" + names
        )
    raise FileNotFoundError(
        "Cannot find the SUHI input file. Put Bengaluru_SUHI_Intensity_2021_2025.csv or .xlsx\n"
        "in the same folder as this script."
    )


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file is missing required columns: " + ", ".join(missing) +
            "\nExpected columns: " + ", ".join(REQUIRED)
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = [
        "urban_mean_LST_C", "urban_valid_pixels", "rural_mean_LST_C",
        "rural_valid_pixels", "SUHI_intensity_C"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=REQUIRED).copy()
    dropped_invalid = before - len(df)

    df = df[
        (df["urban_valid_pixels"] >= MIN_VALID_PIXELS) &
        (df["rural_valid_pixels"] >= MIN_VALID_PIXELS)
    ].copy()
    df["calculated_contrast_C"] = df["urban_mean_LST_C"] - df["rural_mean_LST_C"]
    max_diff = float((df["calculated_contrast_C"] - df["SUHI_intensity_C"]).abs().max())
    if max_diff > 0.02:
        raise ValueError(
            f"SUHI formula validation failed: maximum discrepancy is {max_diff:.4f} °C. "
            "Expected urban_mean_LST_C - rural_mean_LST_C to equal SUHI_intensity_C."
        )

    duplicated_ids = int(df["product_id"].duplicated().sum())
    if duplicated_ids:
        raise ValueError(
            f"Found {duplicated_ids} repeated product_id rows. Resolve repeated acquisitions before uncertainty analysis; "
            "the script will not silently count an acquisition more than once."
        )

    df["SUHI_intensity_C"] = df["calculated_contrast_C"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid acquisitions remain after data-quality filtering.")

    df.attrs["dropped_invalid_rows"] = dropped_invalid
    df.attrs["max_formula_discrepancy"] = max_diff
    return df


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values, [2.5, 97.5]).astype(float))


def iid_bootstrap(values: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[tuple[float, float], tuple[float, float]]:
    n = len(values)
    means = np.empty(reps, dtype=float)
    medians = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
        medians[i] = np.median(sample)
    return percentile_ci(means), percentile_ci(medians)


def moving_block_bootstrap_mean_ci(
    values: np.ndarray, block_size: int, reps: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Non-circular moving-block bootstrap over chronological acquisition order."""
    n = len(values)
    if n < block_size:
        raise ValueError(f"Need at least {block_size} acquisitions for block bootstrap.")
    possible_starts = np.arange(0, n - block_size + 1)
    n_blocks = math.ceil(n / block_size)
    boot_means = np.empty(reps, dtype=float)
    for i in range(reps):
        starts = rng.choice(possible_starts, size=n_blocks, replace=True)
        sample = np.concatenate([values[s:s + block_size] for s in starts])[:n]
        boot_means[i] = sample.mean()
    return percentile_ci(boot_means)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    input_path = locate_input()
    df = load_input(input_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = df["SUHI_intensity_C"].to_numpy(dtype=float)
    n = len(x)
    mean_ci_iid, median_ci_iid = iid_bootstrap(
        x, N_BOOTSTRAPS, np.random.default_rng(RANDOM_SEED)
    )

    block_rows = []
    for j, block_size in enumerate(BLOCK_SIZES):
        ci = moving_block_bootstrap_mean_ci(
            x, block_size, N_BOOTSTRAPS,
            np.random.default_rng(RANDOM_SEED + 100 + j)
        )
        block_rows.append({
            "bootstrap_method": "moving_block_chronological",
            "block_length_acquisitions": block_size,
            "mean_contrast_C": float(np.mean(x)),
            "ci_95_lower_C": ci[0],
            "ci_95_upper_C": ci[1],
            "interval_below_zero": bool(ci[1] < 0),
        })

    yearly = (
        df.groupby("year")["SUHI_intensity_C"]
        .agg(acquisitions="count", mean_C="mean", median_C="median", std_C="std", min_C="min", max_C="max")
        .reset_index()
    )
    yearly.to_csv(OUTPUT_DIR / "SUHI_Uncertainty_By_Year.csv", index=False)

    monthly = (
        df.groupby("month")["SUHI_intensity_C"]
        .agg(acquisitions="count", mean_C="mean", median_C="median", std_C="std")
        .reindex(range(1, 13))
        .reset_index()
    )
    monthly.to_csv(OUTPUT_DIR / "SUHI_Uncertainty_By_Month.csv", index=False)

    positives = int((x > 0).sum())
    negatives = int((x < 0).sum())
    zeros = int((x == 0).sum())
    pos_ci = wilson_interval(positives, n)
    neg_ci = wilson_interval(negatives, n)

    summary_rows = [
        {"method": "IID percentile bootstrap", "block_length_acquisitions": 1,
         "estimate": float(np.mean(x)), "ci_95_lower": mean_ci_iid[0], "ci_95_upper": mean_ci_iid[1]},
        *[{"method": r["bootstrap_method"], "block_length_acquisitions": r["block_length_acquisitions"],
           "estimate": r["mean_contrast_C"], "ci_95_lower": r["ci_95_lower_C"], "ci_95_upper": r["ci_95_upper_C"]}
          for r in block_rows],
        {"method": "IID percentile bootstrap median", "block_length_acquisitions": 1,
         "estimate": float(np.median(x)), "ci_95_lower": median_ci_iid[0], "ci_95_upper": median_ci_iid[1]},
    ]
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "SUHI_Confidence_Intervals.csv", index=False)

    gap_days = df["date"].diff().dt.days.dropna()
    lines = [
        "BENGALURU URBAN–CROPLAND THERMAL CONTRAST: UNCERTAINTY ANALYSIS",
        "=" * 72,
        f"Input file: {input_path.name}",
        f"Valid acquisitions: {n}",
        f"Unique product IDs: {df['product_id'].nunique()}",
        f"Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"Rows dropped for invalid/missing required fields: {df.attrs['dropped_invalid_rows']}",
        f"Maximum formula discrepancy: {df.attrs['max_formula_discrepancy']:.8f} °C",
        f"Minimum valid pixels per zone: {MIN_VALID_PIXELS}",
        "",
        "OVERALL DESCRIPTIVE SUMMARY",
        f"Mean urban-minus-cropland contrast: {np.mean(x):.4f} °C",
        f"Median: {np.median(x):.4f} °C",
        f"Standard deviation: {np.std(x, ddof=1):.4f} °C",
        f"Minimum / maximum: {np.min(x):.4f} / {np.max(x):.4f} °C",
        f"Positive acquisitions: {positives} ({positives/n:.1%})",
        f"Negative acquisitions: {negatives} ({negatives/n:.1%})",
        f"Zero acquisitions: {zeros}",
        f"Wilson 95% interval for positive proportion: [{pos_ci[0]:.3f}, {pos_ci[1]:.3f}]",
        f"Wilson 95% interval for negative proportion: [{neg_ci[0]:.3f}, {neg_ci[1]:.3f}]",
        "",
        "MEAN CONTRAST: 95% RESAMPLING INTERVALS",
        f"IID bootstrap: [{mean_ci_iid[0]:.4f}, {mean_ci_iid[1]:.4f}] °C",
    ]
    for r in block_rows:
        lines.append(
            f"Moving-block bootstrap, block={r['block_length_acquisitions']} acquisitions: "
            f"[{r['ci_95_lower_C']:.4f}, {r['ci_95_upper_C']:.4f}] °C"
        )
    lines += [
        f"IID bootstrap median interval: [{median_ci_iid[0]:.4f}, {median_ci_iid[1]:.4f}] °C",
        "",
        "ACQUISITION TIMING",
        f"Median gap between acquisitions: {gap_days.median():.1f} days" if len(gap_days) else "Median gap: unavailable",
        f"Maximum gap between acquisitions: {gap_days.max():.1f} days" if len(gap_days) else "Maximum gap: unavailable",
        "",
        "INTERPRETATION AND LIMITATIONS",
        "The IID bootstrap treats acquisitions as independent and may understate uncertainty if nearby dates are temporally dependent.",
        "The moving-block bootstrap resamples contiguous chronological groups of acquisitions to retain some short-range dependence.",
        "Block lengths of 3, 5 and 7 acquisitions are sensitivity checks, not a definitive model of temporal dependence.",
        "These intervals describe uncertainty across the retained daytime Landsat acquisitions under the fixed reference zones.",
        "They are not spatial confidence intervals, do not establish causality, and do not measure near-surface air-temperature UHI.",
        "Monthly and annual estimates are descriptive because sampling is uneven across months and years.",
        "A confidence interval entirely below zero would indicate a consistently negative mean under that resampling design; it would not prove Bengaluru has no atmospheric UHI.",
    ]
    summary_path = OUTPUT_DIR / "SUHI_Uncertainty_Summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # Distribution plot with the mean and IID bootstrap CI annotated.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins=18, edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(x), linestyle="--", linewidth=2, label=f"Mean = {np.mean(x):.3f} °C")
    ax.axvline(0, linestyle=":", linewidth=1.8, label="No contrast (0 °C)")
    ax.axvspan(mean_ci_iid[0], mean_ci_iid[1], alpha=0.18, label="IID bootstrap 95% CI")
    ax.set_title("Distribution of Urban–Cropland Surface Temperature Contrast")
    ax.set_xlabel("Urban mean LST − Cropland mean LST (°C)")
    ax.set_ylabel("Number of Landsat acquisitions")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "SUHI_Contrast_Distribution_Uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("SUHI UNCERTAINTY ANALYSIS COMPLETED")
    print("-" * 60)
    print(f"Input: {input_path.name}")
    print(f"Valid acquisitions: {n}")
    print(f"Mean contrast: {np.mean(x):.4f} °C")
    print(f"Median contrast: {np.median(x):.4f} °C")
    print(f"IID bootstrap 95% CI: [{mean_ci_iid[0]:.4f}, {mean_ci_iid[1]:.4f}] °C")
    for r in block_rows:
        print(f"Moving block {r['block_length_acquisitions']} 95% CI: [{r['ci_95_lower_C']:.4f}, {r['ci_95_upper_C']:.4f}] °C")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Read SUHI_Uncertainty_Summary.txt before interpreting the intervals.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
