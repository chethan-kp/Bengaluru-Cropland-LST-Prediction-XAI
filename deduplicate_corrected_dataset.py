from pathlib import Path
import pandas as pd

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_UHI_Corrected_Combined_2021_2025.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {INPUT_FILE.name}"
    )

df = pd.read_csv(INPUT_FILE)

print("ORIGINAL CORRECTED DATASET")
print("-" * 45)
print(f"Rows: {len(df):,}")

# Find exact duplicate records.
exact_duplicate_mask = df.duplicated(keep="first")
duplicate_copies = df.loc[exact_duplicate_mask].copy()

# Save the redundant copies separately for traceability.
duplicate_file = FOLDER / "Bengaluru_UHI_Exact_Duplicate_Copies.csv"
duplicate_copies.to_csv(duplicate_file, index=False)

# Remove only rows identical across every column.
clean = df.drop_duplicates(keep="first").copy()

print("\nEXACT DUPLICATE CHECK")
print("-" * 45)
print(
    "Rows belonging to exact-duplicate groups:",
    int(df.duplicated(keep=False).sum())
)
print("Extra exact copies removed:", len(duplicate_copies))
print(f"Rows remaining: {len(clean):,}")

# Check for repeated date-location pairs after exact deduplication.
keys = ["date", "longitude", "latitude"]

repeated_location_mask = clean.duplicated(
    subset=keys,
    keep=False
)

repeated_locations = clean.loc[repeated_location_mask].copy()

print("\nDATE-LOCATION CHECK")
print("-" * 45)
print(
    "Rows in repeated date-location groups:",
    len(repeated_locations)
)

if not repeated_locations.empty:
    repeated_file = (
        FOLDER / "Bengaluru_UHI_Repeated_Locations_Nonidentical.csv"
    )
    repeated_locations.to_csv(repeated_file, index=False)
    print("Saved repeated locations for manual inspection.")
else:
    print("No repeated date-location groups remain.")

# Report missing values and index limits.
print("\nFINAL DATA QUALITY")
print("-" * 45)
print("Missing values by column:")
print(clean.isna().sum().to_string())

print("\nNDVI range:")
print(clean["NDVI"].min(), "to", clean["NDVI"].max())

print("\nNDMI range:")
print(clean["NDMI"].min(), "to", clean["NDMI"].max())

print("\nRows by year:")
print(clean.groupby("year").size().to_string())

# Save the new dataset and per-date counts.
clean_file = FOLDER / "Bengaluru_UHI_Deduplicated_2021_2025.csv"
clean.to_csv(clean_file, index=False)

counts = (
    clean.groupby(["year", "date"])
    .size()
    .reset_index(name="sample_rows")
    .sort_values(["year", "date"])
)

counts.to_csv(
    FOLDER / "Bengaluru_UHI_Deduplicated_Counts_By_Date.csv",
    index=False
)

print("\nFILES CREATED")
print("-" * 45)
print(clean_file.name)
print("Bengaluru_UHI_Exact_Duplicate_Copies.csv")
print("Bengaluru_UHI_Deduplicated_Counts_By_Date.csv")

print("\nOriginal combined data remains unchanged.")
print("No rows with different measurements were automatically removed.")