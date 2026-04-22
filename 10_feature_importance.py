# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 10: Feature Importance — What Drives Solar Irradiance Predictions?
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Inputs           : outputs/rf_tuned_model.joblib (or rf_model.joblib)
                   outputs/X_train.csv
Outputs          : outputs/feature_importance_bar.png
                   outputs/feature_importance.csv

Academic Context:
  Feature importance from Random Forests (Mean Decrease in Impurity, MDI)
  quantifies the average reduction in node impurity (MSE for regression)
  attributable to each feature across all trees in the forest.

  This analysis serves three research purposes:
  1. PHYSICAL VALIDATION: Important features should align with domain
     knowledge (e.g., Cloud_Fraction should strongly inhibit irradiance).
     If lag features dominate, it validates the autoregressive design.
  2. FEATURE SELECTION: Low-importance features can be pruned to create
     a parsimonious model for deployment in resource-constrained IoT
     devices in smart grid edge nodes.
  3. PEER REVIEW: Reviewers expect justification of model structure.
     "Feature X has 23% importance" is a publishable finding.

  Caveat (for paper discussion section):
  MDI importance can overestimate the importance of high-cardinality
  features. For confirmation, consider SHAP (SHapley Additive exPlanations)
  as an additional robustness check (shap library, optional).
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

OUTPUT_DIR    = "outputs"
TUNED_MODEL   = os.path.join(OUTPUT_DIR, "rf_tuned_model.joblib")
BASE_MODEL    = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
X_TRAIN_CSV   = os.path.join(OUTPUT_DIR, "X_train.csv")
OUTPUT_PLOT   = os.path.join(OUTPUT_DIR, "feature_importance_bar.png")
OUTPUT_CSV    = os.path.join(OUTPUT_DIR, "feature_importance.csv")


def load_model_and_data():
    # Prefer tuned model; fall back to base model
    model_path = TUNED_MODEL if os.path.exists(TUNED_MODEL) else BASE_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ No model found. Run 07_random_forest.py first.")

    model   = joblib.load(model_path)
    X_train = pd.read_csv(X_TRAIN_CSV, parse_dates=["Date"], index_col="Date")

    model_type = "Tuned" if model_path == TUNED_MODEL else "Base"
    print(f"✅ Loaded {model_type} Random Forest model")
    print(f"✅ Feature matrix: {X_train.shape}")
    return model, X_train, model_type


def compute_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extracts MDI feature importances and computes standard deviations
    across individual trees for uncertainty quantification.
    """
    # Mean importance (across all trees)
    importances = model.feature_importances_

    # Std deviation across trees (measures stability/uncertainty)
    std_devs = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )

    importance_df = pd.DataFrame({
        "Feature"    : feature_names,
        "Importance" : importances,
        "Std_Dev"    : std_devs,
        "Importance_%" : importances / importances.sum() * 100,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    importance_df["Rank"] = importance_df.index + 1
    importance_df["Cumulative_%"] = importance_df["Importance_%"].cumsum()

    return importance_df


def create_feature_groups(importance_df: pd.DataFrame) -> pd.Series:
    """
    Assigns each feature to a logical research category for color-coding.
    """
    def categorize(name: str) -> str:
        lag_keywords  = ["Lag", "Roll"]
        time_keywords = ["Month", "DOY", "Day_of_Year", "Season"]
        weather_keys  = ["Temperature", "Humidity", "Cloud", "Wind",
                         "Precipitation", "Relative"]

        if any(k in name for k in lag_keywords):
            return "Autoregressive\n(Lag/Rolling)"
        elif any(k in name for k in time_keywords):
            return "Temporal\n(Cyclical)"
        elif any(k in name for k in weather_keys):
            return "Meteorological\n(Weather)"
        else:
            return "Other"

    return importance_df["Feature"].apply(categorize)


def plot_feature_importance(importance_df: pd.DataFrame, model_type: str):
    """
    Publication-quality horizontal bar chart of feature importances.
    Color-coded by feature category with error bars for tree-level variance.
    """
    print("\n  Generating Feature Importance Bar Chart...")

    # Color palette per category
    category_colors = {
        "Autoregressive\n(Lag/Rolling)"  : "#2563EB",
        "Temporal\n(Cyclical)"           : "#10B981",
        "Meteorological\n(Weather)"      : "#F59E0B",
        "Other"                          : "#94A3B8",
    }

    df = importance_df.copy()
    df["Category"] = create_feature_groups(df)
    df["Color"]    = df["Category"].map(category_colors)

    # Sort ascending for horizontal bar chart (most important at top)
    df_plot = df.sort_values("Importance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df) * 0.45)),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#F8FAFC")

    # ── LEFT: Horizontal Bar Chart ──────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F8FAFC")

    bars = ax.barh(
        df_plot["Feature"],
        df_plot["Importance"],
        xerr       = df_plot["Std_Dev"],
        color      = df_plot["Color"],
        edgecolor  = "white",
        linewidth  = 0.6,
        capsize    = 3,
        error_kw   = {"elinewidth": 1.2, "alpha": 0.7, "ecolor": "#475569"},
        height     = 0.7,
    )

    # Annotate bars with percentage
    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        ax.text(
            row["Importance"] + row["Std_Dev"] + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{row['Importance_%']:.1f}%",
            va="center", ha="left", fontsize=9, color="#1E293B",
        )

    ax.set_xlabel("Mean Decrease in Impurity (MDI) — Feature Importance Score",
                  fontsize=11)
    ax.set_title(
        f"Feature Importance — Random Forest ({model_type} Model)\n"
        "Solar Irradiance Prediction | Error bars = ±1 Std Dev (across trees)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=cat.replace("\n", " "))
                      for cat, c in category_colors.items()
                      if cat in df_plot["Category"].values]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.85, title="Feature Category", title_fontsize=9)

    # ── RIGHT: Cumulative Importance Plot ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8FAFC")

    df_cum = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    ax2.plot(range(1, len(df_cum)+1), df_cum["Cumulative_%"],
             color="#2563EB", linewidth=2.2, marker="o", markersize=5,
             markerfacecolor="white", markeredgewidth=1.5)

    ax2.axhline(y=80, color="#F59E0B", linestyle="--", linewidth=1.2, alpha=0.8,
                label="80% threshold")
    ax2.axhline(y=95, color="#EF4444", linestyle="--", linewidth=1.2, alpha=0.8,
                label="95% threshold")

    # Mark the 80% cutoff point
    n_for_80 = (df_cum["Cumulative_%"] >= 80).idxmax() + 1
    ax2.axvline(x=n_for_80, color="#F59E0B", linestyle=":", linewidth=1, alpha=0.6)
    ax2.text(n_for_80 + 0.1, 35,
             f"{n_for_80} features\nexplain 80%",
             color="#92400E", fontsize=8.5)

    ax2.set_xlabel("Top-N Features", fontsize=10)
    ax2.set_ylabel("Cumulative Importance (%)", fontsize=10)
    ax2.set_title("Cumulative\nImportance", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.35, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Feature importance plot saved → {OUTPUT_PLOT}")


def print_importance_report(importance_df: pd.DataFrame):
    """Prints a formatted importance table suitable for a paper's Table 2."""
    print("\n" + "=" * 65)
    print("  FEATURE IMPORTANCE TABLE (suitable for Table 2 in paper)")
    print("=" * 65)

    total_features = len(importance_df)
    print(f"\n  {'Rank':<5} {'Feature':<28} {'Importance':>12} {'%':>8} {'Cumul%':>9}")
    print(f"  {'-' * 65}")

    for _, row in importance_df.iterrows():
        print(f"  {int(row['Rank']):<5} {row['Feature']:<28} "
              f"{row['Importance']:>12.6f} "
              f"{row['Importance_%']:>7.2f}% "
              f"{row['Cumulative_%']:>8.2f}%")

    # Category summary
    importance_df["Category"] = create_feature_groups(importance_df)
    category_summary = importance_df.groupby("Category")["Importance_%"].sum()

    print(f"\n  ── By Category ──")
    for cat, pct in category_summary.sort_values(ascending=False).items():
        print(f"  {cat.replace(chr(10), ' '):<30} : {pct:.2f}% total importance")


def main():
    print("=" * 65)
    print("  SCRIPT 10: Feature Importance Analysis")
    print("=" * 65)

    model, X_train, model_type = load_model_and_data()
    feature_names = list(X_train.columns)

    importance_df = compute_importance(model, feature_names)

    print_importance_report(importance_df)
    plot_feature_importance(importance_df, model_type)

    # Save to CSV for supplementary data
    importance_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✅ Importance table saved → {OUTPUT_CSV}")
    print("\n  Script 10 complete. Run 11_actual_vs_predicted.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
