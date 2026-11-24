from pathlib import Path
import pandas as pd

# Folder containing the combined dataset
FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Bengaluru_UHI_Combined_2021_2025.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Cannot find Bengaluru_UHI_Combined_2021_2025.csv"
    )

# Load the original combined dataset
df = pd.read_csv(INPUT_FILE)

print("ORIGINAL DATASET")
print("-" * 45)
print(f"Rows: {len(df):,}")

# Convert relevant columns to numeric
numeric_columns = [
    "year",
    "longitude",
    "latitude",
    "LST_C",
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
    "cloud_cover_percent",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# --------------------------------------------------
# STEP 1: Remove only fully identical duplicate rows
# --------------------------------------------------

duplicate_mask = df.duplicated(keep="first")

duplicate_copies = df.loc[duplicate_mask].copy()
deduplicated = df.drop_duplicates(keep="first").copy()

duplicate_copies.to_csv(
    FOLDER / "Bengaluru_UHI_Excluded_Duplicate_Copies.csv",
    index=False
)

print("\nEXACT DUPLICATES")
print("-" * 45)
print(f"Extra duplicate copies removed: {len(duplicate_copies):,}")
print(f"Rows after deduplication: {len(deduplicated):,}")

# --------------------------------------------------
# STEP 2: Identify invalid normalized-difference indices
# --------------------------------------------------

bad_ndvi = (
    deduplicated["NDVI"].isna()
    | ~deduplicated["NDVI"].between(-1, 1)
)

bad_ndmi = (
    deduplicated["NDMI"].isna()
    | ~deduplicated["NDMI"].between(-1, 1)
)

invalid_index_mask = bad_ndvi | bad_ndmi

invalid_rows = deduplicated.loc[invalid_index_mask].copy()

# Document why each row was excluded
def reason_for_exclusion(row):
    reasons = []

    if pd.isna(row["NDVI"]) or not -1 <= row["NDVI"] <= 1:
        reasons.append("NDVI missing or outside [-1, 1]")

    if pd.isna(row["NDMI"]) or not -1 <= row["NDMI"] <= 1:
        reasons.append("NDMI missing or outside [-1, 1]")

    return "; ".join(reasons)


if not invalid_rows.empty:
    invalid_rows["exclusion_reason"] = invalid_rows.apply(
        reason_for_exclusion, axis=1
    )

invalid_rows.to_csv(
    FOLDER / "Bengaluru_UHI_Excluded_Invalid_Indices.csv",
    index=False
)

clean = deduplicated.loc[~invalid_index_mask].copy()

print("\nINVALID INDEX VALUES")
print("-" * 45)
print(f"Rows excluded for invalid indices: {len(invalid_rows):,}")
print(f"Rows remaining: {len(clean):,}")

# --------------------------------------------------
# STEP 3: Run post-cleaning checks
# --------------------------------------------------

print("\nFINAL QUALITY CHECKS")
print("-" * 45)

print("Missing values by column:")
print(clean.isna().sum().to_string())

print("\nRemaining exact duplicate rows:",
      int(clean.duplicated().sum()))

remaining_location_duplicates = clean.duplicated(
    subset=["date", "longitude", "latitude"],
    keep=False
)

print(
    "Rows in repeated date-location groups:",
    int(remaining_location_duplicates.sum())
)

print("\nIndex ranges:")
print("NDVI minimum:", clean["NDVI"].min())
print("NDVI maximum:", clean["NDVI"].max())
print("NDMI minimum:", clean["NDMI"].min())
print("NDMI maximum:", clean["NDMI"].max())

print("\nRows by year:")
print(clean.groupby("year").size().to_string())

# --------------------------------------------------
# STEP 4: Save the cleaned dataset and report
# --------------------------------------------------

clean_path = FOLDER / "Bengaluru_UHI_Clean_2021_2025.csv"
clean.to_csv(clean_path, index=False)

report_path = FOLDER / "Bengaluru_UHI_Cleaning_Report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("BENGALURU UHI DATA CLEANING REPORT\n")
    report.write("=" * 40 + "\n\n")
    report.write(f"Original rows: {len(df):,}\n")
    report.write(f"Exact duplicate copies removed: {len(duplicate_copies):,}\n")
    report.write(f"Invalid-index rows excluded: {len(invalid_rows):,}\n")
    report.write(f"Cleaned rows: {len(clean):,}\n\n")
    report.write("Rows by year:\n")
    report.write(clean.groupby("year").size().to_string())
    report.write("\n\nMissing values by column:\n")
    report.write(clean.isna().sum().to_string())
    report.write("\n\nIndex ranges:\n")
    report.write(f"NDVI: {clean['NDVI'].min()} to {clean['NDVI'].max()}\n")
    report.write(f"NDMI: {clean['NDMI'].min()} to {clean['NDMI'].max()}\n")

print("\nFILES CREATED")
print("-" * 45)
print("Bengaluru_UHI_Clean_2021_2025.csv")
print("Bengaluru_UHI_Excluded_Duplicate_Copies.csv")
print("Bengaluru_UHI_Excluded_Invalid_Indices.csv")
print("Bengaluru_UHI_Cleaning_Report.txt")

print("\nCleaning complete. Original files were preserved.")