# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 04: Feature Engineering
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Input            : outputs/nasa_power_cleaned.csv
Output           : outputs/nasa_power_features.csv

New Features Created:
  Temporal Features (encode cyclical time information):
    - Month        : Calendar month (1–12)
    - Day_of_Year  : Day of year (1–366)
    - Season       : Meteorological season (0=Winter, 1=Pre-Monsoon, etc.)
    - Month_sin/cos: Cyclical sine/cosine encoding of month
    - DOY_sin/cos  : Cyclical sine/cosine encoding of day-of-year

  Lag Features (autoregressive inputs):
    - Solar_Lag_1  : Solar irradiance from t-1 (yesterday)
    - Solar_Lag_2  : Solar irradiance from t-2 (two days ago)
    - Solar_Lag_7  : Solar irradiance from t-7 (same day last week)
    - Solar_Roll7  : 7-day rolling mean of solar irradiance
    - Solar_Roll30 : 30-day rolling mean of solar irradiance

Academic Justification:
  Cyclical encoding (sin/cos) is preferred over raw integer months/days
  because tree-based models (RF, XGBoost) treat December (12) and January
  (1) as far apart, whereas they are seasonally adjacent. Sin/cos encoding
  preserves this cyclical proximity.

  Lag features provide autoregressive context, allowing the model to learn
  from recent historical patterns — critical for short-term forecasting.
=============================================================================
"""

import pandas as pd
import numpy as np
import os

INPUT_CSV  = os.path.join("outputs", "nasa_power_cleaned.csv")
OUTPUT_CSV = os.path.join("outputs", "nasa_power_features.csv")

TARGET_COL = "Solar_Irradiance_MJ_m2"


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"❌ Cleaned CSV not found: '{INPUT_CSV}'. "
            "Run 02_data_preprocessing.py first."
        )
    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and encodes time-based features from the DatetimeIndex.

    Season Mapping (Indian Meteorological Department classification):
        Winter       : Dec, Jan, Feb      → 0
        Pre-Monsoon  : Mar, Apr, May      → 1
        Monsoon      : Jun, Jul, Aug, Sep → 2
        Post-Monsoon : Oct, Nov           → 3
    """
    print("\n  Adding Temporal Features...")

    df["Month"]       = df.index.month
    df["Day_of_Year"] = df.index.dayofyear

    # Season mapping (IMD classification)
    season_map = {
        12: 0, 1: 0, 2: 0,       # Winter
        3:  1, 4: 1, 5: 1,       # Pre-Monsoon
        6:  2, 7: 2, 8: 2, 9: 2, # Monsoon
        10: 3, 11: 3,             # Post-Monsoon
    }
    df["Season"] = df["Month"].map(season_map)

    season_labels = {0: "Winter", 1: "Pre-Monsoon", 2: "Monsoon", 3: "Post-Monsoon"}
    df["Season_Label"] = df["Season"].map(season_labels)

    # Cyclical encoding — prevents the "December-January distance" problem
    # sin/cos transform maps 12 months onto a unit circle
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOY_sin"]   = np.sin(2 * np.pi * df["Day_of_Year"] / 365.25)
    df["DOY_cos"]   = np.cos(2 * np.pi * df["Day_of_Year"] / 365.25)

    print(f"    ✅ Added: Month, Day_of_Year, Season, Season_Label")
    print(f"    ✅ Added: Month_sin, Month_cos, DOY_sin, DOY_cos (cyclical encoding)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates autoregressive lag features from the target variable.

    Lag-1, Lag-2: Short-term memory — useful for clear-sky/cloudy transitions
    Lag-7       : Weekly pattern — same weekday effect, atmosphere patterns
    Roll-7      : Smoothed short-term trend
    Roll-30     : Monthly climatological baseline
    """
    print("\n  Adding Lag Features...")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in DataFrame.")

    df["Solar_Lag_1"]  = df[TARGET_COL].shift(1)
    df["Solar_Lag_2"]  = df[TARGET_COL].shift(2)
    df["Solar_Lag_7"]  = df[TARGET_COL].shift(7)
    df["Solar_Roll7"]  = df[TARGET_COL].shift(1).rolling(window=7).mean()
    df["Solar_Roll30"] = df[TARGET_COL].shift(1).rolling(window=30).mean()

    # NOTE: We shift(1) before rolling to prevent data leakage —
    # the rolling window should only use past data, not today's value.

    print(f"    ✅ Added: Solar_Lag_1 (t-1)")
    print(f"    ✅ Added: Solar_Lag_2 (t-2)")
    print(f"    ✅ Added: Solar_Lag_7 (t-7, weekly)")
    print(f"    ✅ Added: Solar_Roll7 (7-day rolling mean, no-leakage)")
    print(f"    ✅ Added: Solar_Roll30 (30-day rolling mean, no-leakage)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DROP NaNs CREATED BY LAGGING
# ─────────────────────────────────────────────────────────────────────────────
def drop_lag_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag and rolling features introduce NaN in the first N rows.
    These rows cannot be used for training and are safely dropped.
    With a 30-day rolling window + 1-day shift, first 30 rows have NaN.
    """
    before = len(df)
    df.dropna(subset=["Solar_Roll30", "Solar_Lag_7"], inplace=True)
    after = len(df)
    dropped = before - after
    print(f"\n  Dropped {dropped} rows with NaN lag values (expected due to window size).")
    print(f"  Dataset size: {before} → {after} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_feature_summary(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  ENGINEERED FEATURE SET SUMMARY")
    print("=" * 65)
    print(f"\n  Total features (excl. target) : {len(df.columns) - 2}")
    print(f"  Dataset shape                 : {df.shape}")
    print(f"  Date range                    : {df.index.min().date()} → {df.index.max().date()}")

    print("\n  ── Feature Preview (first 3 rows) ──")
    print(df.head(3).to_string())

    print(f"\n  ── Season Distribution ──")
    season_dist = df["Season_Label"].value_counts()
    for season, count in season_dist.items():
        pct = count / len(df) * 100
        print(f"    {season:<15} : {count:>4} days ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SCRIPT 04: Feature Engineering")
    print("=" * 65)

    df = load_data()
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = drop_lag_nans(df)
    print_feature_summary(df)

    df.to_csv(OUTPUT_CSV, index=True, index_label="Date")
    print(f"\n  ✅ Feature-engineered data saved → {OUTPUT_CSV}")
    print("\n  Script 04 complete. Run 05_train_test_split.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
