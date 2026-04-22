# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 06: Persistence (Naive) Baseline Model
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Input            : outputs/y_train.csv, outputs/y_test.csv
Output           : outputs/baseline_predictions.csv
                   outputs/baseline_metrics.txt

The Persistence Model:
  ŷ(t) = y(t-1)

  This is the simplest possible forecast: assume today's solar irradiance
  will be identical to yesterday's. It requires no training, no parameters,
  and no machine learning.

Research Justification for Using a Baseline:
  Every ML paper MUST demonstrate that its model outperforms a naive baseline.
  Without this comparison, reviewers cannot assess whether the complexity of
  the ML model is actually warranted.

  "A model that cannot beat persistence has no practical value."
  — Lorenz et al. (2004), Solar Energy, 77(3), 273-280.

  If our Random Forest achieves MAE = 1.2 MJ/m²/day and the persistence
  model achieves MAE = 2.8 MJ/m²/day, we can claim a ~57% improvement —
  a compelling finding for publication.

  Standard baselines in solar forecasting literature:
    1. Persistence (used here) — y(t) = y(t-1)
    2. Climatological mean   — y(t) = monthly mean irradiance
    3. ARIMA                 — statistical time-series model
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = "outputs"
Y_TRAIN_CSV   = os.path.join(OUTPUT_DIR, "y_train.csv")
Y_TEST_CSV    = os.path.join(OUTPUT_DIR, "y_test.csv")
X_TEST_CSV    = os.path.join(OUTPUT_DIR, "X_test.csv")
OUTPUT_PRED   = os.path.join(OUTPUT_DIR, "baseline_predictions.csv")
OUTPUT_METRICS= os.path.join(OUTPUT_DIR, "baseline_metrics.txt")


def load_data():
    for path in [Y_TRAIN_CSV, Y_TEST_CSV, X_TEST_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ File not found: '{path}'. Run 05_train_test_split.py first."
            )

    y_train = pd.read_csv(Y_TRAIN_CSV, parse_dates=["Date"], index_col="Date").squeeze()
    y_test  = pd.read_csv(Y_TEST_CSV,  parse_dates=["Date"], index_col="Date").squeeze()
    X_test  = pd.read_csv(X_TEST_CSV,  parse_dates=["Date"], index_col="Date")

    print(f"✅ Loaded: Training target ({len(y_train)} samples), Test target ({len(y_test)} samples)")
    return y_train, y_test, X_test


def persistence_forecast(y_train: pd.Series, y_test: pd.Series) -> pd.Series:
    """
    Implements the persistence (naive) model:
        ŷ(t) = y(t-1)

    For the first test day, we use the last training observation as the
    'yesterday' value. This avoids the boundary condition problem.

    Args:
        y_train: Training target series (to get the last training value)
        y_test : Test target series (actual values to forecast)

    Returns:
        pd.Series: Naive predictions aligned with y_test index
    """
    # The full series: last training value + test values
    # Shift test by 1 and fill the first NaN with the last training value
    persistence_preds = y_test.shift(1)
    persistence_preds.iloc[0] = y_train.iloc[-1]   # Boundary condition
    return persistence_preds


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Computes MAE, RMSE, and R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("=" * 65)
    print("  SCRIPT 06: Persistence (Naive) Baseline Model")
    print("=" * 65)

    y_train, y_test, X_test = load_data()

    # Generate persistence forecast
    y_pred_baseline = persistence_forecast(y_train, y_test)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_baseline)

    print("\n" + "=" * 65)
    print("  PERSISTENCE MODEL RESULTS")
    print("=" * 65)
    print(f"\n  Model Type   : Persistence (Naive Baseline)")
    print(f"  Formula      : ŷ(t) = y(t-1)")
    print(f"  Test Period  : {y_test.index.min().date()} → {y_test.index.max().date()}")
    print(f"  Test Samples : {len(y_test)}")

    print(f"\n  ┌─ Evaluation Metrics ─────────────────────────────────┐")
    print(f"  │  MAE  (Mean Absolute Error)      : {metrics['MAE']:>8.4f} MJ/m²/day │")
    print(f"  │  RMSE (Root Mean Squared Error)  : {metrics['RMSE']:>8.4f} MJ/m²/day │")
    print(f"  │  R²   (Coefficient of Determn.)  : {metrics['R2']:>8.4f}             │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Interpretation
    print(f"\n  📊 Interpretation:")
    print(f"     MAE = {metrics['MAE']:.4f} → On average, the persistence model")
    print(f"     is wrong by {metrics['MAE']:.2f} MJ/m²/day.")
    print(f"\n     R² = {metrics['R2']:.4f} → The model explains only")
    print(f"     {metrics['R2']*100:.1f}% of variance — very weak predictive power.")
    print(f"\n     ⚠️  Our ML model MUST beat these numbers to be publishable.")
    print(f"     Target: MAE < {metrics['MAE']:.2f}, R² > {metrics['R2']:.4f}")

    # Save predictions
    result_df = pd.DataFrame({
        "Date"       : y_test.index,
        "Actual"     : y_test.values,
        "Predicted"  : y_pred_baseline.values,
        "Error"      : (y_test.values - y_pred_baseline.values),
        "Abs_Error"  : np.abs(y_test.values - y_pred_baseline.values),
    }).set_index("Date")
    result_df.to_csv(OUTPUT_PRED)

    # Save metrics to text file
    with open(OUTPUT_METRICS, "w") as f:
        f.write("PERSISTENCE BASELINE MODEL METRICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"MAE  : {metrics['MAE']:.6f}\n")
        f.write(f"RMSE : {metrics['RMSE']:.6f}\n")
        f.write(f"R2   : {metrics['R2']:.6f}\n")

    print(f"\n  ✅ Predictions saved → {OUTPUT_PRED}")
    print(f"  ✅ Metrics saved    → {OUTPUT_METRICS}")
    print("\n  Script 06 complete. Run 07_random_forest.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
