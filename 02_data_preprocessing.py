# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 02: Data Preprocessing & Cleaning (Multi-City)
=============================================================================
Input  : outputs/cities/{city}_raw.csv
Output : outputs/cities/{city}_cleaned.csv
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from cities import CITIES

CITIES_DIR = os.path.join("outputs", "cities")

def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df

def handle_missing(df):
    before = df.isnull().sum().sum()
    df.ffill(inplace=True)
    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)
    after = df.isnull().sum().sum()
    print(f"  Missing before: {before} → after: {after}")
    return df

def validate_date_continuity(df):
    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    missing  = expected.difference(df.index)
    if len(missing) > 0:
        df = df.reindex(expected)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        print(f"  Filled {len(missing)} date gap(s).")
    else:
        print(f"  No date gaps detected.")
    return df

def main():
    section("S.U.R.A.J. — Multi-City Preprocessing")

    for city_name in CITIES:
        print(f"\n  Processing: {city_name}...")

        input_csv  = os.path.join(CITIES_DIR, f"{city_name.lower()}_raw.csv")
        output_csv = os.path.join(CITIES_DIR, f"{city_name.lower()}_cleaned.csv")

        try:
            df = load_data(input_csv)
            df = handle_missing(df)
            df = validate_date_continuity(df)
            df.to_csv(output_csv, index=True, index_label="Date")
            print(f"  [OK] Saved → {output_csv}")

        except Exception as e:
            print(f"  [ERROR] {city_name}: {e}")

    print("\n" + "=" * 65)
    print("  All cities preprocessed. Run 03_eda.py next.")
    print("=" * 65)

if __name__ == "__main__":
    main()