# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 05: Temporal Train-Test Split
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Input            : outputs/nasa_power_features.csv
Outputs          : outputs/X_train.csv, X_test.csv, y_train.csv, y_test.csv

Split Strategy   : Temporal (Chronological) Split
  Training Set   : 2019-01-01 → 2022-12-31 (first 4 years)
  Test Set       : 2023-01-01 → 2023-12-31 (final 1 year)

═══════════════════════════════════════════════════════════════════════════════
WHY RANDOM SPLITTING IS INVALID FOR TIME-SERIES (Research Paper Justification)
═══════════════════════════════════════════════════════════════════════════════

Standard train_test_split(random_state=42) shuffles data randomly, which
violates a fundamental assumption of temporal data:

  [PROBLEM 1 — DATA LEAKAGE / LOOKAHEAD BIAS]
  In a random split, the training set may contain data from 2023 while the
  test set contains data from 2019. This means lag features (Solar_Lag_1,
  Solar_Lag_7) in the test set would be computed from data that the model
  had already "seen" during training — artificially inflating all metrics
  (MAE, RMSE, R²). This is NOT a valid evaluation of real-world forecasting.

  [PROBLEM 2 — AUTOCORRELATION DESTRUCTION]
  Daily solar irradiance is strongly autocorrelated (r ≈ 0.7–0.9 with t-1).
  A random split destroys this temporal structure. Models trained on shuffled
  data learn statistical artifacts rather than true meteorological patterns,
  making them useless for actual deployment in energy grid applications.

  [PROBLEM 3 — VIOLATION OF I.I.D. ASSUMPTION]
  Random cross-validation assumes samples are Independent and Identically
  Distributed (i.i.d.). Time-series data is explicitly NOT i.i.d. —
  yesterday's weather directly causes today's weather. Cross-validation
  on shuffled time-series data produces optimistically biased estimates.

  [CORRECT APPROACH — Chronological / Walk-Forward Split]
  The model is trained only on past data and evaluated only on future data
  it has never seen. This mirrors real-world deployment where you must
  forecast the future using only historical information.

  For research papers, use:
    - Temporal hold-out split (used here)
    - Time-Series Cross-Validation (TimeSeriesSplit in scikit-learn)
    - Walk-forward validation (for production systems)

  Reference: Hyndman & Athanasopoulos (2021), "Forecasting: Principles
             and Practice" (3rd ed.), OTexts. Chapter 5.8.
=============================================================================
"""

import pandas as pd
import numpy as np
import os

INPUT_CSV  = os.path.join("outputs", "nasa_power_features.csv")
OUTPUT_DIR = "outputs"

TARGET_COL = "Solar_Irradiance_MJ_m2"
SPLIT_DATE = "2023-01-01"   # First day of the test period

# Columns to exclude from features (metadata / non-numeric)
EXCLUDE_COLS = ["Season_Label", TARGET_COL]


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"❌ Features CSV not found: '{INPUT_CSV}'. "
            "Run 04_feature_engineering.py first."
        )
    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def temporal_split(df: pd.DataFrame):
    """
    Performs a strict chronological train-test split.
    All training data precedes all test data — no overlap.
    """
    split_ts = pd.Timestamp(SPLIT_DATE)

    train_df = df[df.index < split_ts]
    test_df  = df[df.index >= split_ts]

    # Construct feature matrix (X) and target vector (y)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def print_split_summary(X_train, X_test, y_train, y_test):
    total = len(X_train) + len(X_test)
    train_pct = len(X_train) / total * 100
    test_pct  = len(X_test)  / total * 100

    print("\n" + "=" * 65)
    print("  TRAIN / TEST SPLIT SUMMARY")
    print("=" * 65)
    print(f"\n  Split Date       : {SPLIT_DATE}")
    print(f"  Total Samples    : {total}")
    print()
    print(f"  ┌─ Training Set ──────────────────────────────────────┐")
    print(f"  │  Samples   : {len(X_train):>5} rows ({train_pct:.1f}%)            │")
    print(f"  │  Date Range: {X_train.index.min().date()} → {X_train.index.max().date()}    │")
    print(f"  │  Features  : {X_train.shape[1]}                              │")
    print(f"  │  Target μ  : {y_train.mean():.3f} MJ/m²/day                   │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print()
    print(f"  ┌─ Test Set ───────────────────────────────────────────┐")
    print(f"  │  Samples   : {len(X_test):>5} rows ({test_pct:.1f}%)             │")
    print(f"  │  Date Range: {X_test.index.min().date()} → {X_test.index.max().date()}    │")
    print(f"  │  Features  : {X_test.shape[1]}                              │")
    print(f"  │  Target μ  : {y_test.mean():.3f} MJ/m²/day                   │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  Feature List ({X_train.shape[1]} features):")
    for i, col in enumerate(X_train.columns, 1):
        print(f"    {i:>2}. {col}")


def main():
    print("=" * 65)
    print("  SCRIPT 05: Temporal Train-Test Split")
    print("=" * 65)

    df = load_data()
    X_train, X_test, y_train, y_test = temporal_split(df)
    print_split_summary(X_train, X_test, y_train, y_test)

    # Save splits
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"))
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"))
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"))
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"))

    print(f"\n  ✅ X_train.csv, X_test.csv, y_train.csv, y_test.csv saved → {OUTPUT_DIR}/")
    print("\n  Script 05 complete. Run 06_baseline_model.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
