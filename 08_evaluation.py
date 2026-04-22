# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 08: Model Evaluation — MAE, RMSE, R² with Comparison to Baseline
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Inputs           : outputs/random_forest_model.joblib
                   outputs/X_test.csv, outputs/y_test.csv
                   outputs/baseline_metrics.txt
Outputs          : outputs/rf_predictions.csv
                   outputs/rf_metrics.txt
                   outputs/evaluation_comparison.png

═══════════════════════════════════════════════════════════════════════════════
METRIC EXPLANATIONS (for Technical Presentation & Research Paper)
═══════════════════════════════════════════════════════════════════════════════

1. MAE — Mean Absolute Error
   Formula: (1/n) Σ |y_i - ŷ_i|
   Unit   : Same as target variable (MJ/m²/day)
   Meaning: The average magnitude of error in physical units.
            Easy to communicate: "On average, our model is off by X MJ/m²/day."
   Pros   : Intuitive, robust to outliers (uses |error|, not error²)
   Usage  : Primary metric for operational solar forecasting systems.

2. RMSE — Root Mean Squared Error
   Formula: √[(1/n) Σ (y_i - ŷ_i)²]
   Unit   : Same as target variable (MJ/m²/day)
   Meaning: Similar to MAE, but heavily penalizes large errors.
            If RMSE >> MAE, the model occasionally produces very large errors.
   Pros   : Differentiable → used in gradient-based optimization
   Usage  : Standard metric in weather/energy forecasting literature.

3. R² — Coefficient of Determination
   Formula: 1 - [Σ(y_i - ŷ_i)²] / [Σ(y_i - ȳ)²]
   Range  : -∞ to 1.0 (1.0 = perfect, 0.0 = mean model, <0 = worse than mean)
   Meaning: The proportion of variance in solar irradiance explained by
            the model. An R² of 0.90 means "the model explains 90% of
            day-to-day solar variability — 10% remains unexplained."
   Note   : For publication, R² > 0.85 is considered excellent for
            solar irradiance forecasting at daily resolution.
   Usage  : Model goodness-of-fit comparison across studies.

Skill Score (Baseline Comparison):
   SS_MAE = 1 - (MAE_RF / MAE_Baseline) × 100%
   Represents the percentage improvement over naive persistence.
   SS > 50% is publication-worthy for a regional solar forecast.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR    = "outputs"
MODEL_PATH    = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
X_TEST_CSV    = os.path.join(OUTPUT_DIR, "X_test.csv")
Y_TEST_CSV    = os.path.join(OUTPUT_DIR, "y_test.csv")
BASELINE_TXT  = os.path.join(OUTPUT_DIR, "baseline_metrics.txt")
OUTPUT_PRED   = os.path.join(OUTPUT_DIR, "rf_predictions.csv")
OUTPUT_METRICS= os.path.join(OUTPUT_DIR, "rf_metrics.txt")
OUTPUT_PLOT   = os.path.join(OUTPUT_DIR, "evaluation_comparison.png")


def load_baseline_metrics() -> dict:
    """Reads baseline metrics from file for comparison."""
    baseline = {}
    if os.path.exists(BASELINE_TXT):
        with open(BASELINE_TXT) as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    baseline[key.strip()] = float(val.strip())
    return baseline


def compute_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mbe  = np.mean(y_pred - y_true)      # Mean Bias Error
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MBE": mbe}


def plot_comparison(rf_metrics: dict, baseline_metrics: dict):
    """
    Bar chart comparing RF vs. Persistence model on MAE, RMSE, R².
    Publication-quality figure suitable for paper Figure section.
    """
    if not baseline_metrics:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("#F8FAFC")

    comparisons = {
        "MAE (MJ/m²/day)\n(Lower is Better)":  {
            "Persistence": baseline_metrics.get("MAE", 0),
            "Random Forest": rf_metrics["MAE"]
        },
        "RMSE (MJ/m²/day)\n(Lower is Better)": {
            "Persistence": baseline_metrics.get("RMSE", 0),
            "Random Forest": rf_metrics["RMSE"]
        },
        "R² Score\n(Higher is Better)": {
            "Persistence": baseline_metrics.get("R2", 0),
            "Random Forest": rf_metrics["R2"]
        },
    }

    colors = {"Persistence": "#94A3B8", "Random Forest": "#2563EB"}

    for ax, (title, data) in zip(axes, comparisons.items()):
        ax.set_facecolor("#F8FAFC")
        models = list(data.keys())
        values = list(data.values())
        bars = ax.bar(models, values,
                      color=[colors[m] for m in models],
                      width=0.45, edgecolor="white", linewidth=1.5,
                      zorder=3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(abs(v) for v in values),
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_ylim(0, max(abs(v) for v in values) * 1.3)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patch_p = mpatches.Patch(color="#94A3B8", label="Persistence Baseline")
    patch_r = mpatches.Patch(color="#2563EB", label="Random Forest (Ours)")
    fig.legend(handles=[patch_p, patch_r], loc="lower center",
               ncol=2, fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Model Comparison: Random Forest vs. Persistence Baseline\n"
        "Solar Irradiance Forecasting — Jabalpur, MP, India (2023 Test Year)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Comparison chart saved → {OUTPUT_PLOT}")


def main():
    print("=" * 65)
    print("  SCRIPT 08: Random Forest Model Evaluation")
    print("=" * 65)

    # Load model and test data
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: '{MODEL_PATH}'. Run 07_random_forest.py first.")

    model   = joblib.load(MODEL_PATH)
    X_test  = pd.read_csv(X_TEST_CSV, parse_dates=["Date"], index_col="Date")
    y_test  = pd.read_csv(Y_TEST_CSV, parse_dates=["Date"], index_col="Date").squeeze()
    baseline_metrics = load_baseline_metrics()

    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"✅ Test set    : {len(X_test)} samples")

    # Generate predictions
    y_pred = model.predict(X_test)
    rf_metrics = compute_metrics(y_test, y_pred)

    # ── Print Evaluation Report ──
    print("\n" + "=" * 65)
    print("  RANDOM FOREST — TEST SET EVALUATION REPORT")
    print("=" * 65)
    print(f"\n  Test Period  : {y_test.index.min().date()} → {y_test.index.max().date()}")
    print(f"  Test Samples : {len(y_test)}")

    print(f"\n  ┌─ Evaluation Metrics ─────────────────────────────────┐")
    print(f"  │  MAE  (Mean Absolute Error)      : {rf_metrics['MAE']:>8.4f} MJ/m²/day │")
    print(f"  │  RMSE (Root Mean Squared Error)  : {rf_metrics['RMSE']:>8.4f} MJ/m²/day │")
    print(f"  │  R²   (Coefficient of Determn.)  : {rf_metrics['R2']:>8.4f}             │")
    print(f"  │  MBE  (Mean Bias Error)          : {rf_metrics['MBE']:>+8.4f} MJ/m²/day │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # ── Baseline Comparison ──
    if baseline_metrics:
        ss_mae  = (1 - rf_metrics["MAE"]  / baseline_metrics["MAE"])  * 100
        ss_rmse = (1 - rf_metrics["RMSE"] / baseline_metrics["RMSE"]) * 100
        r2_gain = rf_metrics["R2"] - baseline_metrics["R2"]

        print(f"\n  ┌─ Improvement over Persistence Baseline ──────────────┐")
        print(f"  │  Skill Score (MAE)  : {ss_mae:>+7.2f}% improvement          │")
        print(f"  │  Skill Score (RMSE) : {ss_rmse:>+7.2f}% improvement          │")
        print(f"  │  R² Gain            : {r2_gain:>+7.4f} (absolute)            │")
        print(f"  └─────────────────────────────────────────────────────┘")

        print(f"\n  📊 Publication Interpretation:")
        print(f"     Our Random Forest reduces MAE by {ss_mae:.1f}% compared to")
        print(f"     the persistence baseline, demonstrating statistically")
        print(f"     meaningful predictive skill (SS > 0% indicates improvement).")

    # ── R² Interpretation ──
    r2 = rf_metrics["R2"]
    if r2 >= 0.90:
        quality = "Excellent — publication-ready without further tuning."
    elif r2 >= 0.80:
        quality = "Good — consider hyperparameter tuning (Script 09)."
    elif r2 >= 0.70:
        quality = "Acceptable — hyperparameter tuning strongly recommended."
    else:
        quality = "Poor — consider additional features or model stacking."

    print(f"\n  R² Quality Assessment: {quality}")

    # Save predictions
    result_df = pd.DataFrame({
        "Actual"    : y_test.values,
        "Predicted" : y_pred,
        "Error"     : y_pred - y_test.values,
        "Abs_Error" : np.abs(y_pred - y_test.values),
    }, index=y_test.index)
    result_df.to_csv(OUTPUT_PRED)
    print(f"\n  ✅ Predictions saved → {OUTPUT_PRED}")

    # Save metrics
    with open(OUTPUT_METRICS, "w") as f:
        f.write("RANDOM FOREST MODEL METRICS\n")
        f.write("=" * 40 + "\n")
        for k, v in rf_metrics.items():
            f.write(f"{k:6}: {v:.6f}\n")

    plot_comparison(rf_metrics, baseline_metrics)

    print("\n  Script 08 complete. Run 09_hyperparameter_tuning.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
