from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_UHI_Deduplicated_2021_2025.csv"
OUTPUT_FILE = FOLDER / "Bengaluru_UHI_Spatial_Split_2021_2025.csv"

BLOCK_SIZE_M = 5000
RANDOM_SEED = 42


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "date", "year", "longitude", "latitude",
        "LST_C", "NDVI", "NDMI",
        "builtup_fraction_500m_2021", "elevation_m"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Input rows:", f"{len(df):,}")

    # Project coordinates into UTM Zone 43N.
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32643", always_xy=True
    )

    east, north = transformer.transform(
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy()
    )

    df["utm_easting_m"] = east
    df["utm_northing_m"] = north

    # Assign locations to 5 km blocks.
    df["block_x"] = np.floor(
        df["utm_easting_m"] / BLOCK_SIZE_M
    ).astype(int)

    df["block_y"] = np.floor(
        df["utm_northing_m"] / BLOCK_SIZE_M
    ).astype(int)

    df["spatial_block"] = (
        df["block_x"].astype(str)
        + "_"
        + df["block_y"].astype(str)
    )

    # Convert block IDs to a genuine NumPy array before shuffling.
    unique_blocks = (
        df["spatial_block"]
        .drop_duplicates()
        .to_numpy(dtype=str)
        .copy()
    )

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_blocks)

    n_blocks = len(unique_blocks)

    if n_blocks < 10:
        raise ValueError(
            f"Only {n_blocks} blocks found. Check the study area."
        )

    # Approximately 70% training, 15% validation, 15% testing.
    train_end = int(n_blocks * 0.70)
    val_end = int(n_blocks * 0.85)

    train_blocks = set(unique_blocks[:train_end])
    val_blocks = set(unique_blocks[train_end:val_end])
    test_blocks = set(unique_blocks[val_end:])

    def assign_split(block):
        if block in train_blocks:
            return "train"
        if block in val_blocks:
            return "validation"
        return "test"

    df["split"] = df["spatial_block"].map(assign_split)

    # Calendar features for possible seasonal modelling.
    dates = pd.to_datetime(df["date"], errors="raise")
    df["month"] = dates.dt.month
    df["day_of_year"] = dates.dt.dayofyear

    angle = 2 * np.pi * df["day_of_year"] / 365.25
    df["doy_sin"] = np.sin(angle)
    df["doy_cos"] = np.cos(angle)

    # Report block and row distribution.
    print("\nSPATIAL BLOCK SUMMARY")
    print("-" * 45)
    print("Block size (m):", BLOCK_SIZE_M)
    print("Unique blocks:", n_blocks)
    print("Training blocks:", len(train_blocks))
    print("Validation blocks:", len(val_blocks))
    print("Test blocks:", len(test_blocks))

    print("\nRows by split:")
    print(df["split"].value_counts().to_string())

    print("\nRows by year and split:")
    print(pd.crosstab(df["year"], df["split"]).to_string())

    # Confirm each block belongs to one split only.
    blocks_per_split = df.groupby(
        "spatial_block"
    )["split"].nunique()

    crossing = int((blocks_per_split > 1).sum())
    print("\nBlocks crossing splits:", crossing)

    if crossing != 0:
        raise RuntimeError("A spatial block appears in multiple splits.")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCreated:", OUTPUT_FILE.name)
    print("No samples were deleted.")


if __name__ == "__main__":
    main()