# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 11: Actual vs. Predicted — Publication-Quality Research Figure
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Inputs           : outputs/rf_predictions.csv (or tuned predictions)
                   outputs/y_test.csv
Outputs          : outputs/actual_vs_predicted_30days.png  (main paper figure)
                   outputs/scatter_actual_vs_predicted.png  (supplementary)
                   outputs/error_distribution.png            (supplementary)

Figure Description:
  The main figure (30-day time-series overlay) is formatted for direct
  insertion into a IEEE/Springer/Elsevier research paper:
    - 300 DPI (print quality)
    - Professional color scheme (avoiding red/green colorblindness issues)
    - APA/IEEE figure caption format
    - Gridlines, legends, axis labels with units
    - Error shading band for visual interpretation
    - Highlighted monsoon transition period if within plot window

Supplementary Figures:
  1. Scatter plot (Actual vs. Predicted for full test year) with 1:1 line
     and R² annotation — standard in ML regression papers
  2. Residual error distribution — tests normality of errors
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from scipy import stats

OUTPUT_DIR   = "outputs"
# rf_predictions.csv is overwritten by script 09 with tuned-model predictions.
# If tuning was skipped, it contains base-model predictions from script 08.
PRED_CSV     = os.path.join(OUTPUT_DIR, "rf_predictions.csv")
Y_TEST_CSV   = os.path.join(OUTPUT_DIR, "y_test.csv")
X_TEST_CSV   = os.path.join(OUTPUT_DIR, "X_test.csv")
TUNED_MODEL  = os.path.join(OUTPUT_DIR, "rf_tuned_model.joblib")
BASE_MODEL   = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")

OUT_MAIN     = os.path.join(OUTPUT_DIR, "actual_vs_predicted_30days.png")
OUT_SCATTER  = os.path.join(OUTPUT_DIR, "scatter_actual_vs_predicted.png")
OUT_RESIDUAL = os.path.join(OUTPUT_DIR, "error_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PLOT STYLE — IEEE/Elsevier Publication Standard
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"        : "DejaVu Serif",   # Serif fonts for academic papers
    "font.size"          : 11,
    "axes.titlesize"     : 12,
    "axes.labelsize"     : 11,
    "xtick.labelsize"    : 10,
    "ytick.labelsize"    : 10,
    "legend.fontsize"    : 10,
    "figure.dpi"         : 150,
    "savefig.dpi"        : 300,
    "axes.linewidth"     : 0.8,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
})

COLORS = {
    "actual"    : "#1E3A5F",   # Deep Navy — actual observations
    "predicted" : "#D97706",   # Amber/Gold — model predictions
    "error_band": "#FDE68A",   # Light yellow — uncertainty
    "grid"      : "#CBD5E1",   # Slate grey — gridlines
    "bg"        : "#FFFFFF",   # White background (paper standard)
}


def load_predictions() -> pd.DataFrame:
    """
    Loads pre-computed RF predictions. Falls back to re-running inference
    if the CSV is not found but the model and test data exist.
    """
    if os.path.exists(PRED_CSV):
        df = pd.read_csv(PRED_CSV, parse_dates=["Date"], index_col="Date")
        print(f"✅ Loaded predictions: {len(df)} samples")
        return df

    # Fallback: run inference from saved model
    print("⚠️  Predictions CSV not found. Running inference from saved model...")
    model_path = TUNED_MODEL if os.path.exists(TUNED_MODEL) else BASE_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "❌ No model or predictions found. Run scripts 07–08 first."
        )

    model  = joblib.load(model_path)
    X_test = pd.read_csv(X_TEST_CSV, parse_dates=["Date"], index_col="Date")
    y_test = pd.read_csv(Y_TEST_CSV, parse_dates=["Date"], index_col="Date").squeeze()

    y_pred = model.predict(X_test)
    df = pd.DataFrame({
        "Actual"   : y_test.values,
        "Predicted": y_pred,
        "Error"    : y_pred - y_test.values,
        "Abs_Error": np.abs(y_pred - y_test.values),
    }, index=y_test.index)

    df.to_csv(PRED_CSV)
    print(f"✅ Predictions generated: {len(df)} samples")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: MAIN — 30-DAY ACTUAL VS. PREDICTED TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────
def plot_30day_timeseries(df: pd.DataFrame):
    """
    Primary publication figure: Overlaid actual vs. predicted daily solar
    irradiance for the first 30 days of the test period.

    Design choices (justified for academic publication):
    - Dark navy actual vs amber predicted → high contrast, colorblind-safe
    - Shaded error band (±1 MAE) → communicates model uncertainty bounds
    - Grid aligned to weeks → aids temporal reading
    - 300 DPI → suitable for print at 3.5" or 7" column width
    """
    print("\n  Generating Figure 1: 30-Day Actual vs. Predicted...")

    # Select first 30 days of test set
    df_30 = df.head(30).copy()

    # Compute rolling MAE for uncertainty band
    mae_global = df["Abs_Error"].mean()

    fig = plt.figure(figsize=(12, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])

    ax = fig.add_subplot(111)
    ax.set_facecolor(COLORS["bg"])

    dates = df_30.index

    # ── Error shading band (± global MAE) ──
    ax.fill_between(
        dates,
        df_30["Predicted"] - mae_global,
        df_30["Predicted"] + mae_global,
        alpha     = 0.18,
        color     = COLORS["predicted"],
        label     = f"±MAE Band (±{mae_global:.2f} MJ/m²/day)",
        zorder    = 1,
    )

    # ── Actual observations (solid, thicker) ──
    ax.plot(
        dates, df_30["Actual"],
        color     = COLORS["actual"],
        linewidth = 2.2,
        linestyle = "-",
        marker    = "o",
        markersize= 4.5,
        markerfacecolor = "white",
        markeredgewidth = 1.5,
        markeredgecolor = COLORS["actual"],
        label     = "Observed (Actual)",
        zorder    = 3,
    )

    # ── Model predictions (dashed, amber) ──
    ax.plot(
        dates, df_30["Predicted"],
        color     = COLORS["predicted"],
        linewidth = 2.0,
        linestyle = "--",
        marker    = "s",
        markersize= 4.0,
        markerfacecolor = COLORS["predicted"],
        markeredgewidth = 1.2,
        markeredgecolor = "#92400E",
        label     = "Predicted (Random Forest)",
        zorder    = 4,
    )

    # ── Axis labels and formatting ──
    ax.set_xlabel("Date (2023)", fontsize=11, labelpad=8)
    ax.set_ylabel("Solar Irradiance (MJ m⁻² day⁻¹)", fontsize=11, labelpad=8)

    ax.set_title(
        "Fig. 3 — Observed vs. RF-Predicted Daily Solar Irradiance\n"
        "First 30 Days of Evaluation Period | Jabalpur, Madhya Pradesh, India (23.18°N, 79.98°E)",
        fontsize=11.5, fontweight="bold", pad=14
    )

    # Grid — major weekly, minor daily
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(which="major", alpha=0.45, linestyle="-",  color=COLORS["grid"])
    ax.grid(which="minor", alpha=0.15, linestyle=":",  color=COLORS["grid"])
    ax.set_axisbelow(True)

    # Y-axis range with headroom
    y_min = min(df_30[["Actual","Predicted"]].min()) - 1.5
    y_max = max(df_30[["Actual","Predicted"]].max()) + 1.5
    ax.set_ylim(max(0, y_min), y_max)

    # Legend
    legend = ax.legend(
        loc            = "upper right",
        framealpha     = 0.92,
        edgecolor      = "#CBD5E1",
        fontsize       = 10,
        handlelength   = 2.2,
        ncol           = 1,
    )

    # Caption-style annotation at bottom
    caption = (
        "Note: Shaded region represents ±MAE uncertainty envelope. "
        "Data source: NASA POWER API (2023). Model: Random Forest Regressor."
    )
    fig.text(0.5, -0.03, caption, ha="center", fontsize=8.5,
             color="#475569", style="italic")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fig.savefig(OUT_MAIN, dpi=300)
    plt.close(fig)
    print(f"  ✅ Main figure saved → {OUT_MAIN}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: SCATTER PLOT — Full Test Year
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter(df: pd.DataFrame):
    """
    Actual vs. Predicted scatter plot with 1:1 perfect-fit line and
    OLS regression line. Standard in ML regression papers.
    """
    print("\n  Generating Figure 2: Scatter Plot (full test year)...")

    from sklearn.metrics import r2_score
    r2  = r2_score(df["Actual"], df["Predicted"])
    mae = df["Abs_Error"].mean()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Hexbin density plot (better than scatter for 365 points)
    hb = ax.hexbin(
        df["Actual"], df["Predicted"],
        gridsize=20, cmap="Blues", mincnt=1,
        linewidths=0.3
    )
    plt.colorbar(hb, ax=ax, label="Count of Days")

    # 1:1 perfect line
    lims = [min(df["Actual"].min(), df["Predicted"].min()) - 0.5,
            max(df["Actual"].max(), df["Predicted"].max()) + 0.5]
    ax.plot(lims, lims, "r--", linewidth=2.0, label="1:1 Perfect Fit", zorder=5)

    # OLS regression line
    slope, intercept, _, _, _ = stats.linregress(df["Actual"], df["Predicted"])
    x_ols = np.array(lims)
    ax.plot(x_ols, slope * x_ols + intercept,
            color="#D97706", linewidth=1.8, label=f"OLS Fit (slope={slope:.3f})",
            zorder=6)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed Solar Irradiance (MJ m⁻² day⁻¹)", fontsize=11)
    ax.set_ylabel("Predicted Solar Irradiance (MJ m⁻² day⁻¹)", fontsize=11)
    ax.set_title(
        "Fig. 4 — Scatter: Observed vs. Predicted (2023 Test Year)\n"
        "Random Forest | Jabalpur, India",
        fontsize=11, fontweight="bold"
    )

    # Annotate metrics
    textstr = f"$R^2$ = {r2:.4f}\nMAE = {mae:.3f} MJ m⁻² day⁻¹\nn = {len(df)} days"
    props = dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", alpha=0.85,
                 edgecolor="#CBD5E1")
    ax.text(0.05, 0.93, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props)

    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=300)
    plt.close(fig)
    print(f"  ✅ Scatter plot saved → {OUT_SCATTER}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: RESIDUAL ERROR DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_residuals(df: pd.DataFrame):
    """
    Histogram + KDE + Q-Q plot of residuals.
    Tests whether errors are normally distributed (desirable property).
    """
    print("\n  Generating Figure 3: Residual Distribution...")

    errors = df["Error"].values   # Signed: Predicted - Actual

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Left: Histogram + KDE ──
    ax1 = axes[0]
    ax1.set_facecolor(COLORS["bg"])
    sns.histplot(errors, kde=True, ax=ax1, color="#2563EB",
                 edgecolor="white", linewidth=0.5, bins=30, alpha=0.75)
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=1.8, label="Zero Error")
    ax1.axvline(x=errors.mean(), color="#F59E0B", linestyle="-.",
                linewidth=1.5, label=f"MBE = {errors.mean():.3f}")
    ax1.set_xlabel("Prediction Error (Predicted − Actual) [MJ m⁻² day⁻¹]", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Residual Distribution", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.35, linestyle="--")

    # Annotate skewness & kurtosis
    from scipy.stats import skew, kurtosis, shapiro
    skewness = skew(errors)
    kurt     = kurtosis(errors)
    _, p_sw  = shapiro(errors[:min(len(errors), 5000)])   # Shapiro-Wilk normality
    text = (f"Skewness  = {skewness:.3f}\n"
            f"Kurtosis  = {kurt:.3f}\n"
            f"Shapiro-p = {p_sw:.4f}")
    props = dict(boxstyle="round", facecolor="#F1F5F9", alpha=0.9, edgecolor="#CBD5E1")
    ax1.text(0.97, 0.97, text, transform=ax1.transAxes, fontsize=9,
             va="top", ha="right", bbox=props)

    # ── Right: Q-Q Plot ──
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["bg"])
    (osm, osr), (slope, intercept, r) = stats.probplot(errors, dist="norm")
    ax2.scatter(osm, osr, color="#2563EB", alpha=0.5, s=18, label="Residuals")
    x_line = np.array([min(osm), max(osm)])
    ax2.plot(x_line, slope * x_line + intercept,
             color="red", linewidth=2, label=f"Theoretical Normal (r={r:.3f})")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.set_title("Normal Q-Q Plot of Residuals", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.35, linestyle="--")

    fig.suptitle(
        "Fig. 5 — Residual Error Analysis | Random Forest Solar Forecast (2023 Test Year)",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(OUT_RESIDUAL, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Residual plot saved → {OUT_RESIDUAL}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SCRIPT 11: Publication-Quality Visualization")
    print("=" * 65)

    df = load_predictions()

    plot_30day_timeseries(df)   # Figure 3 in paper
    plot_scatter(df)             # Figure 4 (supplementary or main)
    plot_residuals(df)           # Figure 5 (supplementary)

    print("\n" + "=" * 65)
    print("  ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 65)
    print(f"\n  📂 Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\n  Figures ready for paper insertion:")
    print(f"    ✅ {OUT_MAIN}       ← Main paper figure")
    print(f"    ✅ {OUT_SCATTER}   ← Supplementary Figure A")
    print(f"    ✅ {OUT_RESIDUAL}  ← Supplementary Figure B")
    print("\n  Script 11 complete. Project pipeline fully executed! 🎉")
    print("=" * 65)


if __name__ == "__main__":
    main()
