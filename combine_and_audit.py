from pathlib import Path
import pandas as pd

# Folder containing this Python file and the five CSVs
DATA_DIR = Path(__file__).resolve().parent

YEARS = [2021, 2022, 2023, 2024, 2025]

EXPECTED_ROWS = {
    2021: 27980,
    2022: 48062,
    2023: 66521,
    2024: 76725,
    2025: 65898,
}

COLUMNS = [
    "date",
    "year",
    "product_id",
    "cloud_cover_percent",
    "longitude",
    "latitude",
    "LST_C",
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
]

NUMERIC_COLUMNS = [
    "year",
    "cloud_cover_percent",
    "longitude",
    "latitude",
    "LST_C",
    "NDVI",
    "NDMI",
    "builtup_fraction_500m_2021",
    "elevation_m",
]


def main():
    datasets = []

    print("READING YEARLY CSV FILES")
    print("-" * 45)

    for year in YEARS:
        filename = f"Bengaluru_Pixel_Samples_{year}.csv"
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find {filename}. "
                "Check that all five CSVs are in this folder."
            )

        df = pd.read_csv(path)

        missing_columns = [
            column for column in COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{filename} is missing columns: {missing_columns}"
            )

        df = df[COLUMNS].copy()

        print(
            f"{year}: {len(df):,} rows "
            f"(expected {EXPECTED_ROWS[year]:,})"
        )

        if len(df) != EXPECTED_ROWS[year]:
            print("  WARNING: Actual row count differs from expected.")

        datasets.append(df)

    # Combine all five years
    data = pd.concat(datasets, ignore_index=True)

    # Convert numerical fields
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Validate dates
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    invalid_date = parsed_dates.isna()

    data["date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    print("\nCOMBINED DATASET")
    print("-" * 45)
    print(f"Total rows: {len(data):,}")
    print(f"Total columns: {len(data.columns)}")

    print("\nRows by year:")
    print(data.groupby("year").size().to_string())

    # Missing-value audit
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    # Find duplicate samples for the same date and location
    duplicate_mask = data.duplicated(
        subset=["date", "longitude", "latitude"],
        keep=False
    )

    # Create screening flags; do not automatically delete data
    flags = pd.DataFrame(index=data.index)

    flags["invalid_date"] = invalid_date
    flags["missing_values"] = data.isna().any(axis=1)

    flags["year_date_mismatch"] = (
        parsed_dates.dt.year != data["year"]
    ).fillna(True)

    flags["duplicate_date_location"] = duplicate_mask

    flags["LST_outside_screening_range"] = (
        (data["LST_C"] < -10) |
        (data["LST_C"] > 80)
    ) | data["LST_C"].isna()

    flags["NDVI_outside_range"] = (
        (data["NDVI"] < -1) |
        (data["NDVI"] > 1)
    ) | data["NDVI"].isna()

    flags["NDMI_outside_range"] = (
        (data["NDMI"] < -1) |
        (data["NDMI"] > 1)
    ) | data["NDMI"].isna()

    builtup = data["builtup_fraction_500m_2021"]

    flags["builtup_fraction_outside_range"] = (
        (builtup < 0) |
        (builtup > 1)
    ) | builtup.isna()

    elevation = data["elevation_m"]

    flags["elevation_outside_screening_range"] = (
        (elevation < -100) |
        (elevation > 3000)
    ) | elevation.isna()

    cloud = data["cloud_cover_percent"]

    flags["cloud_cover_outside_range"] = (
        (cloud < 0) |
        (cloud > 100)
    ) | cloud.isna()

    print("\nQUALITY AUDIT")
    print("-" * 45)
    print("Duplicate date-location rows:", int(duplicate_mask.sum()))

    print("\nRows flagged by each check:")
    print(flags.sum().to_string())

    # Save combined data without deleting flagged records
    combined_path = DATA_DIR / "Bengaluru_UHI_Combined_2021_2025.csv"
    data.to_csv(combined_path, index=False)

    # Save sample counts by date
    counts = (
        data.groupby(["year", "date"])
        .size()
        .reset_index(name="sample_rows")
        .sort_values(["year", "date"])
    )

    counts.to_csv(
        DATA_DIR / "Bengaluru_UHI_Sample_Counts_By_Date.csv",
        index=False
    )

    # Save rows that need inspection
    flagged_mask = flags.any(axis=1)

    if flagged_mask.any():
        flagged = data.loc[flagged_mask].copy()

        flagged["audit_issues"] = flags.loc[flagged_mask].apply(
            lambda row: "; ".join(row.index[row.to_numpy()]),
            axis=1
        )

        flagged.to_csv(
            DATA_DIR / "Bengaluru_UHI_Flagged_Rows.csv",
            index=False
        )

    # Save audit summary
    report_path = DATA_DIR / "Bengaluru_UHI_Audit_Summary.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("BENGALURU UHI DATASET AUDIT\n")
        report.write("=" * 35 + "\n\n")
        report.write(f"Total rows: {len(data):,}\n\n")
        report.write("Rows by year:\n")
        report.write(data.groupby("year").size().to_string())
        report.write("\n\nMissing values by column:\n")
        report.write(data.isna().sum().to_string())
        report.write("\n\nRows flagged by each check:\n")
        report.write(flags.sum().to_string())
        report.write(
            f"\n\nDuplicate date-location rows: "
            f"{int(duplicate_mask.sum())}\n"
        )

    print("\nFILES CREATED")
    print("-" * 45)
    print("Bengaluru_UHI_Combined_2021_2025.csv")
    print("Bengaluru_UHI_Sample_Counts_By_Date.csv")
    print("Bengaluru_UHI_Audit_Summary.txt")

    if flagged_mask.any():
        print("Bengaluru_UHI_Flagged_Rows.csv")

    print("\nAudit finished. No records were automatically deleted.")


if __name__ == "__main__":
    main()