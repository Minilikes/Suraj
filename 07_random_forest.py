# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 07: Random Forest Regressor — Model Training
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Inputs           : outputs/X_train.csv, outputs/y_train.csv
Outputs          : outputs/random_forest_model.joblib
                   outputs/training_log.txt

Model Choice Justification (for Research Paper):
  Random Forest Regression is selected over deep learning models because:

  1. INTERPRETABILITY: Feature importance scores provide physical insight
     into which meteorological variables drive solar irradiance — critical
     for publishing in energy systems journals.

  2. SAMPLE EFFICIENCY: Deep learning (LSTM, Transformer) requires thousands
     of samples to generalize; RF achieves high accuracy with ~1500 daily
     records (our 4-year training set).

  3. ROBUSTNESS TO OUTLIERS: The ensemble averaging mechanism (majority
     vote over 200 trees) is naturally robust to extreme values caused
     by dust storms, fog, or data sensor errors.

  4. NO HYPERPARAMETER SENSITIVITY: Unlike neural networks, RF does not
     require learning rate tuning, batch size selection, or gradient clipping.

  5. TRAINING SPEED: RF trains in seconds on CPU; LSTM requires GPU hours.
     This enables rapid iteration during research.

  Reference: Breiman (2001), "Random Forests", Machine Learning, 45(1), 5-32.
             Tyralis et al. (2019), Hydrology, 6(2), 28. (solar forecasting)
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT_DIR    = "outputs"
X_TRAIN_CSV   = os.path.join(OUTPUT_DIR, "X_train.csv")
Y_TRAIN_CSV   = os.path.join(OUTPUT_DIR, "y_train.csv")
MODEL_PATH    = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
LOG_PATH      = os.path.join(OUTPUT_DIR, "training_log.txt")

RANDOM_STATE  = 42   # Fixed for reproducibility across paper revisions


def load_training_data():
    for path in [X_TRAIN_CSV, Y_TRAIN_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ File not found: '{path}'. Run 05_train_test_split.py first."
            )

    X_train = pd.read_csv(X_TRAIN_CSV, parse_dates=["Date"], index_col="Date")
    y_train = pd.read_csv(Y_TRAIN_CSV, parse_dates=["Date"], index_col="Date").squeeze()

    print(f"✅ Training data loaded:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   Features     : {list(X_train.columns)}")
    return X_train, y_train


def build_model() -> RandomForestRegressor:
    """
    Constructs the Random Forest Regressor with research-justified parameters.

    n_estimators=200:
        200 trees provides a strong bias-variance trade-off for datasets of
        this size (~1500 samples). Beyond 500 trees, accuracy gains are
        marginal while computation grows linearly.

    max_depth=None:
        Trees grow fully unless constrained by min_samples_split. This
        allows the model to capture complex non-linear meteorological
        interactions (e.g., temperature × cloud fraction interaction).

    min_samples_split=5:
        A node must have at least 5 samples to be split, preventing
        overfitting to noisy individual days.

    max_features='sqrt':
        Each split considers √(n_features) candidate features — the
        standard RF configuration that ensures tree decorrelation.

    n_jobs=-1:
        Uses all available CPU cores for parallel tree construction.

    random_state=42:
        Ensures bit-exact reproducibility across all runs, a requirement
        for scientific publication (Method section: "All experiments use
        random_state=42 for reproducibility").
    """
    model = RandomForestRegressor(
        n_estimators    = 200,
        max_depth       = None,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features    = "sqrt",
        bootstrap       = True,
        oob_score       = True,       # Out-of-bag score (free validation)
        n_jobs          = -1,
        random_state    = RANDOM_STATE,
        verbose         = 0,
    )
    return model


def train_model(model: RandomForestRegressor, X_train, y_train):
    """Trains the model and logs timing information."""

    print("\n" + "=" * 65)
    print("  TRAINING RANDOM FOREST REGRESSOR")
    print("=" * 65)
    print(f"  Configuration:")
    print(f"    n_estimators      : {model.n_estimators}")
    print(f"    max_depth         : {model.max_depth} (unlimited)")
    print(f"    max_features      : {model.max_features}")
    print(f"    min_samples_split : {model.min_samples_split}")
    print(f"    min_samples_leaf  : {model.min_samples_leaf}")
    print(f"    OOB Score         : {model.oob_score}")
    print(f"    Random State      : {model.random_state}")
    print(f"    n_jobs            : {model.n_jobs} (all CPU cores)")
    print()
    print(f"  Training on {len(X_train)} samples with {X_train.shape[1]} features...")
    print(f"  Please wait...")

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    print(f"\n  ✅ TRAINING COMPLETE!")
    print(f"  Training time : {elapsed:.2f} seconds")

    # Out-of-Bag evaluation (free in-training validation)
    oob_score = model.oob_score_
    print(f"  OOB R² Score  : {oob_score:.4f}")
    print(f"  (OOB score is a free, unbiased estimate of generalization performance)")

    # In-sample performance (sanity check — should be near-perfect for RF)
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2  = r2_score(y_train, y_train_pred)
    print(f"\n  In-Sample Training Metrics (sanity check):")
    print(f"    Train MAE : {train_mae:.4f} MJ/m²/day")
    print(f"    Train R²  : {train_r2:.4f}")
    print(f"  (High train R² + good OOB R² indicates proper fitting, not overfitting)")

    return model, elapsed


def save_model(model: RandomForestRegressor, elapsed: float, X_train):
    """Persists the trained model to disk using joblib serialization."""
    joblib.dump(model, MODEL_PATH)
    print(f"\n  ✅ Model saved → {MODEL_PATH}")

    # Write training log
    log_content = (
        f"RANDOM FOREST TRAINING LOG\n"
        f"{'=' * 40}\n"
        f"Training Date   : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Random State    : {model.random_state}\n"
        f"n_estimators    : {model.n_estimators}\n"
        f"max_depth       : {model.max_depth}\n"
        f"max_features    : {model.max_features}\n"
        f"Training Samples: {model.n_features_in_}\n"
        f"Features Used   : {list(X_train.columns)}\n"
        f"OOB R² Score    : {model.oob_score_:.6f}\n"
        f"Training Time   : {elapsed:.2f} seconds\n"
    )

    with open(LOG_PATH, "w") as f:
        f.write(log_content)
    print(f"  ✅ Training log saved → {LOG_PATH}")


def main():
    print("=" * 65)
    print("  SCRIPT 07: Random Forest — Model Training")
    print("=" * 65)

    X_train, y_train = load_training_data()
    model = build_model()
    model, elapsed = train_model(model, X_train, y_train)
    save_model(model, elapsed, X_train)

    print("\n  Script 07 complete. Run 08_evaluation.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
