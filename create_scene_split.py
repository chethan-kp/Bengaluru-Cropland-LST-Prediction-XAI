
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

FOLDER = Path(__file__).resolve().parent

INPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Deduplicated_2021_2025.csv"
)

OUTPUT_FILE = (
    FOLDER / "Bengaluru_UHI_Scene_Split_By_Year_2021_2025.csv"
)


# ==================================================
# 2. LOAD AND VALIDATE
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    required = ["date", "product_id", "LST_C"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    data["product_id"] = (
        data["product_id"].astype("string").str.strip()
    )

    if data[["date", "product_id", "LST_C"]].isna().any().any():
        raise ValueError(
            "Missing date, product ID, or LST values found."
        )

    if data["product_id"].eq("").any():
        raise ValueError("Blank product IDs found.")

    data["date_key"] = data["date"].dt.strftime("%Y-%m-%d")
    data["year"] = data["date"].dt.year

    # Verify each acquisition has one product ID and date.
    scenes = data[
        ["product_id", "date_key", "year"]
    ].drop_duplicates()

    if scenes["product_id"].duplicated().any():
        raise ValueError(
            "A product ID maps to multiple dates or years."
        )

    if scenes["date_key"].duplicated().any():
        raise ValueError(
            "Multiple product IDs map to the same date. "
            "Group these acquisitions together before splitting."
        )

    print("YEAR-STRATIFIED SCENE SPLIT")
    print("-" * 60)
    print(f"Input rows: {len(data):,}")
    print(f"Unique scenes: {len(scenes)}")


    # ==================================================
    # 3. SPLIT SCENES WITHIN EACH YEAR
    # ==================================================

    scene_assignments = {}

    for year, year_scenes in scenes.groupby("year"):

        products = year_scenes["product_id"].to_numpy().copy()

        rng = np.random.default_rng(42 + int(year))
        rng.shuffle(products)

        number_of_scenes = len(products)

        if number_of_scenes < 3:
            raise ValueError(
                f"Year {year} has too few scenes to create "
                "train, validation, and test partitions."
            )

        # Approximately 15% each for validation and test,
        # with at least one scene in each partition.
        number_test = max(
            1, int(round(number_of_scenes * 0.15))
        )

        number_validation = max(
            1, int(round(number_of_scenes * 0.15))
        )

        # Ensure at least one training scene remains.
        while (
            number_test + number_validation
            >= number_of_scenes
        ):
            if number_validation > 1:
                number_validation -= 1
            elif number_test > 1:
                number_test -= 1
            else:
                raise ValueError(
                    f"Cannot split scenes for {year}."
                )

        test_products = products[:number_test]

        validation_products = products[
            number_test:
            number_test + number_validation
        ]

        train_products = products[
            number_test + number_validation:
        ]

        for product in train_products:
            scene_assignments[product] = "train"

        for product in validation_products:
            scene_assignments[product] = "validation"

        for product in test_products:
            scene_assignments[product] = "test"

        print(
            f"{year}: {number_of_scenes} scenes -> "
            f"{len(train_products)} train, "
            f"{len(validation_products)} validation, "
            f"{len(test_products)} test"
        )

    data["scene_split"] = (
        data["product_id"]
        .map(scene_assignments)
    )

    if data["scene_split"].isna().any():
        raise ValueError(
            "Some scenes were not assigned to a partition."
        )


    # ==================================================
    # 4. VERIFY NO SCENE OR DATE LEAKAGE
    # ==================================================

    products_crossing = (
        data.groupby("product_id")["scene_split"].nunique()
    )

    dates_crossing = (
        data.groupby("date_key")["scene_split"].nunique()
    )

    product_violations = int(
        (products_crossing > 1).sum()
    )

    date_violations = int(
        (dates_crossing > 1).sum()
    )

    if product_violations or date_violations:
        raise ValueError(
            "Scene or date leakage detected."
        )

    print("\nROWS BY SCENE SPLIT")
    print("-" * 60)

    print(
        data.groupby("scene_split")
        .agg(
            rows=("scene_split", "size"),
            scenes=("product_id", "nunique"),
            dates=("date_key", "nunique"),
        )
        .to_string()
    )

    print("\nSCENES BY YEAR AND SPLIT")
    print("-" * 60)

    print(
        pd.crosstab(
            scenes.assign(
                scene_split=scenes["product_id"].map(
                    scene_assignments
                )
            )["year"],
            scenes["product_id"].map(scene_assignments),
        ).to_string()
    )

    print("\nROWS BY YEAR AND SPLIT")
    print("-" * 60)

    print(
        pd.crosstab(
            data["year"], data["scene_split"]
        ).to_string()
    )

    print(f"\nProduct IDs crossing splits: {product_violations}")
    print(f"Dates crossing splits: {date_violations}")


    # ==================================================
    # 5. SAVE A NEW DATASET
    # ==================================================

    data = data.drop(columns=["date_key"])

    data["date"] = pd.to_datetime(data["date"]).dt.strftime(
        "%Y-%m-%d"
    )

    data.to_csv(OUTPUT_FILE, index=False)

    print("\nCreated:")
    print(OUTPUT_FILE.name)

    print("\nYear-stratified scene split completed.")


if __name__ == "__main__":
    main()
