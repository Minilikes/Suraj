# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 09: Hyperparameter Tuning — RandomizedSearchCV
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Inputs           : outputs/X_train.csv, outputs/y_train.csv
Outputs          : outputs/rf_tuned_model.joblib
                   outputs/tuning_results.csv
                   outputs/tuned_metrics.txt

Strategy: RandomizedSearchCV
  We use RandomizedSearchCV over GridSearchCV because:
  - The parameter space is large (3 parameters × multiple values = 100+ combos)
  - RandomizedSearch evaluates n_iter=30 random combinations in a fraction
    of GridSearch's time while achieving ~95% of optimal performance
  - Time-series validation uses TimeSeriesSplit (NOT standard k-fold CV)
    to prevent lookahead bias during cross-validation

  Reference: Bergstra & Bengio (2012), "Random Search for Hyper-Parameter
             Optimization", JMLR, 13, 281-305.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import time
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

warnings.filterwarnings("ignore")

OUTPUT_DIR     = "outputs"
X_TRAIN_CSV    = os.path.join(OUTPUT_DIR, "X_train.csv")
Y_TRAIN_CSV    = os.path.join(OUTPUT_DIR, "y_train.csv")
X_TEST_CSV     = os.path.join(OUTPUT_DIR, "X_test.csv")
Y_TEST_CSV     = os.path.join(OUTPUT_DIR, "y_test.csv")
TUNED_MODEL    = os.path.join(OUTPUT_DIR, "rf_tuned_model.joblib")
TUNING_CSV     = os.path.join(OUTPUT_DIR, "tuning_results.csv")
TUNED_METRICS  = os.path.join(OUTPUT_DIR, "tuned_metrics.txt")
TUNED_PRED     = os.path.join(OUTPUT_DIR, "rf_predictions.csv")  # overwrite with best model

RANDOM_STATE   = 42
N_ITER         = 30    # Number of random parameter combinations to try
CV_SPLITS      = 5     # TimeSeriesSplit folds


def load_data():
    for p in [X_TRAIN_CSV, Y_TRAIN_CSV, X_TEST_CSV, Y_TEST_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ Missing: '{p}'. Run earlier scripts first.")

    X_train = pd.read_csv(X_TRAIN_CSV, parse_dates=["Date"], index_col="Date")
    y_train = pd.read_csv(Y_TRAIN_CSV, parse_dates=["Date"], index_col="Date").squeeze()
    X_test  = pd.read_csv(X_TEST_CSV,  parse_dates=["Date"], index_col="Date")
    y_test  = pd.read_csv(Y_TEST_CSV,  parse_dates=["Date"], index_col="Date").squeeze()

    print(f"✅ Data loaded: Train={len(X_train)} | Test={len(X_test)}")
    return X_train, y_train, X_test, y_test


def define_search_space() -> dict:
    """
    Defines the hyperparameter search space with research justification:

    n_estimators [100–500]:
        More trees → lower variance, but diminishing returns after 300.
        Upper bound 500 to keep wall-clock time feasible.

    max_depth [None, 10–30]:
        None = fully grown trees (higher variance, lower bias).
        Shallow trees (10) = regularization → better generalization.

    min_samples_split [2–15]:
        Controls minimum samples needed to split a node.
        Higher values → simpler trees (more regularization).

    min_samples_leaf [1–8]:
        Leaf size regularization. Prevents single-sample leaves that
        memorize noisy training days.

    max_features ['sqrt', 'log2', 0.5]:
        Controls feature subsetting per split.
        'sqrt' ≈ √14 ≈ 3–4 features — standard RF default.
        0.5 = 50% of features — more expressive per split.

    bootstrap [True, False]:
        Whether to use bootstrap sampling. True → bagging (standard RF).
        False → all samples per tree → typically higher variance.
    """
    param_dist = {
        "n_estimators"      : randint(100, 501),
        "max_depth"         : [None, 10, 15, 20, 25, 30],
        "min_samples_split" : randint(2, 16),
        "min_samples_leaf"  : randint(1, 9),
        "max_features"      : ["sqrt", "log2", 0.5, 0.7],
        "bootstrap"         : [True, False],
    }
    return param_dist


def run_search(X_train, y_train, param_dist: dict):
    """
    Executes RandomizedSearchCV with TimeSeriesSplit cross-validation.

    TimeSeriesSplit ensures each validation fold only uses data from
    a period AFTER the training fold — mimicking real forward deployment.
    This is the correct CV strategy for time-series ML in research papers.
    """
    print("\n" + "=" * 65)
    print("  HYPERPARAMETER SEARCH (RandomizedSearchCV + TimeSeriesSplit)")
    print("=" * 65)
    print(f"  n_iter          : {N_ITER} random combinations")
    print(f"  CV Strategy     : TimeSeriesSplit(n_splits={CV_SPLITS})")
    print(f"  Scoring Metric  : Negative MAE (neg_mean_absolute_error)")
    print(f"  Random State    : {RANDOM_STATE}")
    print(f"\n  ⏳ This may take 1–5 minutes depending on CPU speed...")

    base_model = RandomForestRegressor(
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = param_dist,
        n_iter              = N_ITER,
        scoring             = "neg_mean_absolute_error",
        cv                  = tscv,
        refit               = True,       # Refit on full train set with best params
        return_train_score  = True,
        random_state        = RANDOM_STATE,
        n_jobs              = -1,
        verbose             = 1,
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    print(f"\n  ✅ Search complete in {elapsed:.1f} seconds")
    return search, elapsed


def print_tuning_results(search: RandomizedSearchCV):
    print("\n" + "=" * 65)
    print("  TUNING RESULTS — BEST HYPERPARAMETERS FOUND")
    print("=" * 65)

    best_params = search.best_params_
    best_cv_mae = -search.best_score_

    print(f"\n  Best CV MAE (TimeSeriesSplit): {best_cv_mae:.4f} MJ/m²/day")
    print(f"\n  Best Hyperparameters:")
    for param, value in sorted(best_params.items()):
        print(f"    {param:<22} : {value}")

    # Save all results to CSV for supplementary material
    results_df = pd.DataFrame(search.cv_results_)
    results_df["mean_test_mae"]  = -results_df["mean_test_score"]
    results_df["mean_train_mae"] = -results_df["mean_train_score"]
    cols = ["rank_test_score", "mean_test_mae", "mean_train_mae",
            "std_test_score", "mean_fit_time"] + \
           [c for c in results_df.columns if c.startswith("param_")]
    results_df = results_df[cols].sort_values("rank_test_score")
    results_df.to_csv(TUNING_CSV, index=False)
    print(f"\n  ✅ All {len(results_df)} results saved → {TUNING_CSV}")

    # Top 5
    print(f"\n  Top 5 Parameter Combinations (by CV MAE):")
    print(f"  {'Rank':<6} {'CV MAE':>10}  {'n_est':>6}  {'max_d':>6}  {'feat':>8}")
    print(f"  {'-'*45}")
    for _, row in results_df.head(5).iterrows():
        n_est = row.get("param_n_estimators", "?")
        max_d = row.get("param_max_depth", "None")
        feat  = row.get("param_max_features", "?")
        print(f"  {int(row['rank_test_score']):<6} {row['mean_test_mae']:>10.4f}  "
              f"{str(n_est):>6}  {str(max_d):>6}  {str(feat):>8}")


def evaluate_tuned_model(model, X_test, y_test):
    """Evaluates the tuned model on hold-out test set."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n" + "=" * 65)
    print("  TUNED MODEL — HOLD-OUT TEST SET METRICS")
    print("=" * 65)
    print(f"\n  ┌─ Final Evaluation Metrics ───────────────────────────┐")
    print(f"  │  MAE  : {mae:>8.4f} MJ/m²/day                       │")
    print(f"  │  RMSE : {rmse:>8.4f} MJ/m²/day                       │")
    print(f"  │  R²   : {r2:>8.4f}                                  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Save tuned model predictions (overwrites base-model rf_predictions.csv
    # so that script 11 automatically uses the best available model)
    y_pred = model.predict(X_test)
    result_df = pd.DataFrame({
        "Actual"    : y_test.values,
        "Predicted" : y_pred,
        "Error"     : y_pred - y_test.values,
        "Abs_Error" : np.abs(y_pred - y_test.values),
    }, index=y_test.index)
    result_df.index.name = "Date"
    result_df.to_csv(TUNED_PRED)
    print(f"  Tuned predictions saved -> {TUNED_PRED}")

    # Save tuned metrics
    with open(TUNED_METRICS, "w") as f:
        f.write("TUNED RANDOM FOREST METRICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"MAE  : {mae:.6f}\n")
        f.write(f"RMSE : {rmse:.6f}\n")
        f.write(f"R2   : {r2:.6f}\n")

    return mae, rmse, r2


def main():
    print("=" * 65)
    print("  SCRIPT 09: Hyperparameter Tuning")
    print("=" * 65)

    X_train, y_train, X_test, y_test = load_data()
    param_dist = define_search_space()
    search, elapsed = run_search(X_train, y_train, param_dist)
    print_tuning_results(search)

    # Evaluate best model on test set
    best_model = search.best_estimator_
    evaluate_tuned_model(best_model, X_test, y_test)

    # Save tuned model
    joblib.dump(best_model, TUNED_MODEL)
    print(f"\n  ✅ Tuned model saved → {TUNED_MODEL}")
    print("\n  Script 09 complete. Run 10_feature_importance.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
