
from pathlib import Path
import csv
import re

# =====================================================
# 1. PROJECT FOLDER
# =====================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "SUHI_Sensitivity_Input_Inventory.txt"

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules"
}

# Relevant keywords for identifying source files
KEYWORDS = [
    "suhi", "lst", "landsat", "worldcover",
    "urban", "rural", "cropland", "pixel",
    "thermal", "st_b10", "qa_pixel",
    "st_qa", "st_cdist", "export.table"
]

IMPORTANT_COLUMNS = [
    "date", "product_id", "longitude", "latitude",
    "lon", "lat", "lst", "lst_c",
    "urban_mean_lst_c", "rural_mean_lst_c",
    "urban_valid_pixels", "rural_valid_pixels",
    "suhi_intensity_c", "land_cover",
    "class", "distance"
]

# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def file_size(path):
    size = path.stat().st_size
    return f"{size / (1024 * 1024):.2f} MB"


def read_csv_header(path):
    try:
        with open(path, "r", encoding="utf-8-sig",
                  errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        return [str(c).strip() for c in header]
    except Exception as e:
        return [f"Could not read CSV header: {e}"]


def read_excel_headers(path):
    results = []

    try:
        from openpyxl import load_workbook

        wb = load_workbook(
            path, read_only=True, data_only=True
        )

        for ws in wb.worksheets:
            headers = next(
                ws.iter_rows(
                    min_row=1,
                    max_row=1,
                    values_only=True
                ),
                ()
            )

            headers = [
                str(c).strip() if c is not None else ""
                for c in headers
            ]

            results.append(
                (ws.title, ws.max_row, headers)
            )

        wb.close()

    except ImportError:
        results.append((
            "Install openpyxl to inspect Excel files",
            0,
            []
        ))

    except Exception as e:
        results.append((f"Error: {e}", 0, []))

    return results


def inspect_source_code(path):
    keywords_found = []

    try:
        if path.stat().st_size > 3 * 1024 * 1024:
            return []

        content = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).lower()

        for keyword in KEYWORDS:
            if keyword.lower() in content:
                keywords_found.append(keyword)

    except Exception:
        pass

    return keywords_found


def matching_columns(columns):
    normalized = [
        re.sub(r"[^a-z0-9_]", "", str(c).lower())
        for c in columns
    ]

    return [
        c for c, n in zip(columns, normalized)
        if any(k in n for k in IMPORTANT_COLUMNS)
    ]


# =====================================================
# 3. SCAN PROJECT FILES
# =====================================================

print("=" * 65)
print("BENGALURU SUHI SENSITIVITY INPUT INVENTORY")
print("=" * 65)

all_files = []

for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    if any(part in SKIP_DIRS for part in path.parts):
        continue

    all_files.append(path)

print(f"\nProject folder: {ROOT}")
print(f"Total files discovered: {len(all_files)}")

report = []

report.append("BENGALURU SUHI SENSITIVITY INPUT INVENTORY")
report.append("=" * 65)
report.append(f"Project folder: {ROOT}")
report.append(f"Total files: {len(all_files)}")

# =====================================================
# 4. INSPECT DATA FILES
# =====================================================

data_extensions = {
    ".csv", ".xlsx", ".xls"
}

source_extensions = {
    ".js", ".py", ".txt"
}

geo_extensions = {
    ".tif", ".tiff", ".vrt",
    ".shp", ".geojson", ".gpkg", ".img"
}

for path in sorted(all_files):

    ext = path.suffix.lower()

    if ext not in data_extensions:
        continue

    relative = path.relative_to(ROOT)

    report.append("\n" + "-" * 65)
    report.append(f"DATA FILE: {relative}")
    report.append(f"Size: {file_size(path)}")

    print(f"\nInspecting: {relative}")

    if ext == ".csv":

        headers = read_csv_header(path)

        report.append("Format: CSV")
        report.append(f"Columns: {headers}")

        matched = matching_columns(headers)

        report.append(
            f"Relevant columns: {matched}"
        )

        if any(
            "urban" in str(c).lower()
            or "rural" in str(c).lower()
            for c in headers
        ):
            report.append(
                "CATEGORY: Possible urban/rural summary data"
            )

        if any(
            "latitude" in str(c).lower()
            or str(c).lower() == "lat"
            for c in headers
        ) and any(
            "longitude" in str(c).lower()
            or str(c).lower() == "lon"
            for c in headers
        ):
            report.append(
                "CATEGORY: Contains geographical coordinates"
            )

        if any(
            "class" in str(c).lower()
            or "landcover" in str(c).lower()
            for c in headers
        ):
            report.append(
                "CATEGORY: Possible land-cover information"
            )

    elif ext == ".xlsx":

        report.append("Format: Excel")

        sheets = read_excel_headers(path)

        for sheet, rows, headers in sheets:

            report.append(
                f"Sheet: {sheet} | Approx. rows: {rows}"
            )

            report.append(
                f"Columns: {headers}"
            )

            report.append(
                f"Relevant columns: {matching_columns(headers)}"
            )

    elif ext == ".xls":

        report.append(
            "Legacy Excel format: header inspection skipped."
        )


# =====================================================
# 5. FIND GOOGLE EARTH ENGINE / PYTHON SCRIPTS
# =====================================================

report.append("\n\n" + "=" * 65)
report.append("SOURCE CODE FILES")
report.append("=" * 65)

for path in sorted(all_files):

    if path.suffix.lower() not in source_extensions:
        continue

    hits = inspect_source_code(path)

    if hits:

        relative = path.relative_to(ROOT)

        report.append("\n" + "-" * 65)
        report.append(f"SOURCE: {relative}")
        report.append(f"Keywords found: {hits}")
        report.append(f"Size: {file_size(path)}")


# =====================================================
# 6. FIND GEOSPATIAL FILES
# =====================================================

report.append("\n\n" + "=" * 65)
report.append("GEOSPATIAL / RASTER FILES")
report.append("=" * 65)

for path in sorted(all_files):

    if path.suffix.lower() in geo_extensions:

        report.append(
            f"{path.relative_to(ROOT)} | {file_size(path)}"
        )


# =====================================================
# 7. SUMMARY
# =====================================================

report.append("\n\n" + "=" * 65)
report.append("IMPORTANT NEXT CHECK")
report.append("=" * 65)

report.append(
    "Look for files containing original pixel-level LST, "
    "coordinates, land-cover classes, distance from city centre, "
    "or the original Google Earth Engine script."
)

report.append(
    "The existing 115-acquisition summary alone cannot "
    "recalculate alternative geographical reference zones."
)

OUTPUT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("SCAN COMPLETED")
print("=" * 65)
print(f"Report created:\n{OUTPUT}")
print("\nOpen the TXT report and share it here.")
