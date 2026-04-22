# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 02: Data Preprocessing & Cleaning
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Input            : outputs/nasa_power_raw.csv
Output           : outputs/nasa_power_cleaned.csv

Operations:
  1. Load raw CSV with proper DatetimeIndex
  2. Audit missing values (counts + percentages)
  3. Handle nulls via forward-fill then linear interpolation
  4. Validate date continuity (no gaps in daily series)
  5. Print cleaned dataset summary
  6. Save cleaned CSV
=============================================================================
"""

import pandas as pd
import numpy as np
import os

INPUT_CSV  = os.path.join("outputs", "nasa_power_raw.csv")
OUTPUT_CSV = os.path.join("outputs", "nasa_power_cleaned.csv")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: SECTION HEADER PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA WITH DATETIME INDEX
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw NASA POWER CSV and sets the 'Date' column as a
    proper DatetimeIndex. This is essential for time-series operations
    such as resampling, rolling windows, and lag feature creation.
    """
    section("STEP 1: Loading Raw Dataset")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Raw CSV not found at '{filepath}'.\n"
            "   Please run 01_data_fetch.py first."
        )

    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")

    # Sort chronologically (NASA API should return sorted data, but verify)
    df.sort_index(inplace=True)

    print(f"  ✅ Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Date Range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Columns    : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: MISSING VALUE AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a formatted missing value report per column.
    Returns a summary DataFrame for logging.
    """
    section("STEP 2: Missing Value Audit")

    total_rows = len(df)
    missing_summary = pd.DataFrame({
        "Missing Count"  : df.isnull().sum(),
        "Missing Percent": (df.isnull().sum() / total_rows * 100).round(2),
        "Dtype"          : df.dtypes,
    })

    print(f"\n  Total rows in dataset: {total_rows}")
    print(f"\n  {'Column':<30} {'Missing':>8} {'  %':>8}  {'Dtype':>10}")
    print(f"  {'-'*60}")
    for col, row in missing_summary.iterrows():
        status = "⚠️ " if row["Missing Count"] > 0 else "✅ "
        print(f"  {status}{col:<28} {int(row['Missing Count']):>8} "
              f"  {row['Missing Percent']:>7.2f}%  {str(row['Dtype']):>10}")

    total_missing = df.isnull().sum().sum()
    print(f"\n  Total missing cells: {total_missing} / {total_rows * len(df.columns)}")

    return missing_summary


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy (justified for meteorological time-series):

    Method 1 — Forward Fill (ffill):
        Replaces NaN with the last valid observation.
        Rationale: Weather variables exhibit strong temporal autocorrelation;
        yesterday's conditions are a sound first approximation of today's.

    Method 2 — Linear Interpolation (fallback):
        If consecutive NaNs remain after ffill (e.g., at the dataset start),
        linear interpolation fills them using surrounding valid values.
        This preserves the monotonic continuity of meteorological trends.

    Reference: Morales-Salinas et al. (2019) — Imputation Methods in
               Meteorological Time Series for Agricultural Applications.
    """
    section("STEP 3: Handling Missing Values")

    before = df.isnull().sum().sum()

    # Pass 1: Forward fill
    df.ffill(inplace=True)
    after_ffill = df.isnull().sum().sum()

    # Pass 2: Linear interpolation (handles NaNs at series start)
    df.interpolate(method="linear", inplace=True)

    # Pass 3: Backward fill safety net (if NaN is at the very start)
    df.bfill(inplace=True)

    after_all = df.isnull().sum().sum()

    print(f"  Before imputation : {before} missing values")
    print(f"  After forward-fill: {after_ffill} missing values")
    print(f"  After interpolation: {after_all} missing values")

    if after_all == 0:
        print("  ✅ All missing values successfully resolved.")
    else:
        print(f"  ⚠️  {after_all} missing values remain — manual inspection needed.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: DATE CONTINUITY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_date_continuity(df: pd.DataFrame):
    """
    Ensures there are no gaps in the daily time-series.
    For time-series ML models, a continuous index is critical because
    lag features and rolling statistics assume uniform spacing.
    """
    section("STEP 4: Date Continuity Validation")

    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    missing_dates  = expected_range.difference(df.index)

    print(f"  Expected daily records: {len(expected_range)}")
    print(f"  Actual records        : {len(df)}")
    print(f"  Missing date gaps     : {len(missing_dates)}")

    if len(missing_dates) > 0:
        print(f"  ⚠️  Missing dates: {missing_dates.tolist()[:10]}")
        # Reindex to full date range, filling any gaps
        df = df.reindex(expected_range)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        print(f"  ✅ Gaps filled via reindex + forward-fill.")
    else:
        print("  ✅ No date gaps detected. Series is continuous.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DATASET SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    section("STEP 5: Cleaned Dataset Summary")

    print(f"\n  Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Index Type : {type(df.index).__name__}")
    print(f"  Date Range : {df.index.min().date()} → {df.index.max().date()}")

    print("\n  ── First 5 Rows ──")
    print(df.head().to_string())

    print("\n  ── Descriptive Statistics ──")
    print(df.describe().round(3).to_string())

    print("\n  ── Data Types ──")
    print(df.dtypes.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_data(INPUT_CSV)
    audit_missing(df)
    df = handle_missing(df)
    df = validate_date_continuity(df)
    print_summary(df)

    # Save cleaned dataset
    df.to_csv(OUTPUT_CSV, index=True, index_label="Date")
    print(f"\n  ✅ Cleaned data saved → {OUTPUT_CSV}")
    print("\n  Script 02 complete. Run 03_eda.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
