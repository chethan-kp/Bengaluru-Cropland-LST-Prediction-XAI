from pathlib import Path
import pandas as pd

FOLDER = Path(__file__).resolve().parent

YEARS = [2021, 2022, 2023, 2024, 2025]

EXPECTED_ROWS = {
    2021: 28151,
    2022: 48061,
    2023: 66520,
    2024: 76724,
    2025: 65895,
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

    print("READING CORRECTED DATASETS")
    print("-" * 50)

    for year in YEARS:
        filename = f"Bengaluru_Pixel_Samples_{year}_Corrected.csv"
        path = FOLDER / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {filename}\n"
                "Download the corrected CSV into this folder."
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
            print("WARNING: Row count differs from the expected count.")

        datasets.append(df)

    # Combine all five years without deleting any rows.
    data = pd.concat(datasets, ignore_index=True)

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    invalid_date = parsed_dates.isna()

    # Keep dates in a consistent format.
    data["date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    print("\nCOMBINED CORRECTED DATASET")
    print("-" * 50)
    print(f"Total rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")

    print("\nRows by year:")
    print(data.groupby("year").size().to_string())

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    # Identify exact duplicate records.
    exact_duplicate = data.duplicated(keep=False)

    # Identify repeated date-location groups.
    date_location_duplicate = data.duplicated(
        subset=["date", "longitude", "latitude"],
        keep=False
    )

    # Flag suspicious values; do not automatically remove them.
    flags = pd.DataFrame(index=data.index)

    flags["invalid_date"] = invalid_date
    flags["missing_values"] = data.isna().any(axis=1)

    flags["year_date_mismatch"] = (
        parsed_dates.dt.year != data["year"]
    ).fillna(True)

    flags["exact_duplicate"] = exact_duplicate
    flags["duplicate_date_location"] = date_location_duplicate

    flags["LST_outside_screening_range"] = (
        (data["LST_C"] < -10) |
        (data["LST_C"] > 80) |
        data["LST_C"].isna()
    )

    flags["NDVI_outside_range"] = (
        (data["NDVI"] < -1) |
        (data["NDVI"] > 1) |
        data["NDVI"].isna()
    )

    flags["NDMI_outside_range"] = (
        (data["NDMI"] < -1) |
        (data["NDMI"] > 1) |
        data["NDMI"].isna()
    )

    builtup = data["builtup_fraction_500m_2021"]

    flags["builtup_fraction_outside_range"] = (
        (builtup < 0) |
        (builtup > 1) |
        builtup.isna()
    )

    elevation = data["elevation_m"]

    flags["elevation_outside_screening_range"] = (
        (elevation < -100) |
        (elevation > 3000) |
        elevation.isna()
    )

    cloud = data["cloud_cover_percent"]

    flags["cloud_cover_outside_range"] = (
        (cloud < 0) |
        (cloud > 100) |
        cloud.isna()
    )

    print("\nQUALITY AUDIT")
    print("-" * 50)
    print("Rows in exact-duplicate groups:", int(exact_duplicate.sum()))
    print(
        "Rows in repeated date-location groups:",
        int(date_location_duplicate.sum())
    )

    print("\nRows flagged by each check:")
    print(flags.sum().to_string())

    print("\nIndex ranges:")
    print("NDVI minimum:", data["NDVI"].min())
    print("NDVI maximum:", data["NDVI"].max())
    print("NDMI minimum:", data["NDMI"].min())
    print("NDMI maximum:", data["NDMI"].max())

    # Save the combined data without automatically deleting records.
    combined_file = FOLDER / "Bengaluru_UHI_Corrected_Combined_2021_2025.csv"
    data.to_csv(combined_file, index=False)

    # Save counts by date.
    counts = (
        data.groupby(["year", "date"])
        .size()
        .reset_index(name="sample_rows")
        .sort_values(["year", "date"])
    )

    counts.to_csv(
        FOLDER / "Bengaluru_UHI_Corrected_Sample_Counts_By_Date.csv",
        index=False
    )

    # Save rows that require inspection.
    flagged_mask = flags.any(axis=1)

    if flagged_mask.any():
        flagged = data.loc[flagged_mask].copy()

        flagged["audit_issues"] = flags.loc[flagged_mask].apply(
            lambda row: "; ".join(row.index[row.to_numpy()]),
            axis=1
        )

        flagged.to_csv(
            FOLDER / "Bengaluru_UHI_Corrected_Flagged_Rows.csv",
            index=False
        )

    # Write the audit report.
    with open(
        FOLDER / "Bengaluru_UHI_Corrected_Audit_Report.txt",
        "w",
        encoding="utf-8"
    ) as report:
        report.write("BENGALURU UHI CORRECTED DATASET AUDIT\n")
        report.write("=" * 45 + "\n\n")
        report.write(f"Total rows: {len(data):,}\n\n")
        report.write("Rows by year:\n")
        report.write(data.groupby("year").size().to_string())
        report.write("\n\nMissing values:\n")
        report.write(data.isna().sum().to_string())
        report.write("\n\nRows flagged by each check:\n")
        report.write(flags.sum().to_string())
        report.write("\n\nIndex ranges:\n")
        report.write(f"NDVI: {data['NDVI'].min()} to {data['NDVI'].max()}\n")
        report.write(f"NDMI: {data['NDMI'].min()} to {data['NDMI'].max()}\n")

    print("\nFILES CREATED")
    print("-" * 50)
    print("Bengaluru_UHI_Corrected_Combined_2021_2025.csv")
    print("Bengaluru_UHI_Corrected_Sample_Counts_By_Date.csv")
    print("Bengaluru_UHI_Corrected_Audit_Report.txt")

    if flagged_mask.any():
        print("Bengaluru_UHI_Corrected_Flagged_Rows.csv")

    print("\nAudit completed. No records were automatically deleted.")


if __name__ == "__main__":
    main()