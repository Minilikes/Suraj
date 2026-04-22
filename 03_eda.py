# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 03: Exploratory Data Analysis (EDA)
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Input            : outputs/nasa_power_cleaned.csv
Outputs          : outputs/eda_correlation_heatmap.png
                   outputs/eda_solar_seasonality.png
                   outputs/eda_feature_distributions.png
                   outputs/eda_solar_monthly_boxplot.png

Analyses:
  1. Correlation heatmap — identifies key predictors of solar irradiance
  2. Seasonal trend line plot — one full year of solar irradiance
  3. Feature distribution histograms — data shape & outliers
  4. Monthly box plot — variance and median by month
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

INPUT_CSV  = os.path.join("outputs", "nasa_power_cleaned.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PLOT STYLE — Publication-Ready Academic Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.4,
    "grid.linestyle"    : "--",
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
})

PALETTE = {
    "primary"   : "#2563EB",   # Royal Blue
    "secondary" : "#F59E0B",   # Amber
    "accent"    : "#10B981",   # Emerald
    "danger"    : "#EF4444",   # Red
    "bg"        : "#F8FAFC",   # Off-white
}


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ Run 02_data_preprocessing.py first. Missing: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"], index_col="Date")
    print(f"✅ Loaded cleaned data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Pearson correlation matrix of all meteorological variables.
    Helps identify multicollinearity and the strongest predictors
    of solar irradiance for feature selection decisions.
    """
    print("\n  Generating Correlation Heatmap...")

    corr = df.corr(numeric_only=True).round(2)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle mask

    cmap = sns.diverging_palette(230, 15, s=85, l=50, as_cmap=True)

    heatmap = sns.heatmap(
        corr,
        annot        = True,
        fmt          = ".2f",
        cmap         = cmap,
        center       = 0,
        vmin         = -1,
        vmax         = 1,
        linewidths   = 0.5,
        linecolor    = "white",
        square       = True,
        annot_kws    = {"size": 10, "weight": "bold"},
        ax           = ax,
        cbar_kws     = {"shrink": 0.75, "label": "Pearson r"},
    )

    ax.set_title(
        "Pearson Correlation Matrix — Solar Irradiance & Meteorological Variables\n"
        "Location: Jabalpur, MP, India (23.18°N, 79.98°E) | 2019–2023",
        fontsize=11, fontweight="bold", pad=15
    )
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    output_path = os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✅ Saved → {output_path}")

    # Print top correlates with solar irradiance
    target = "Solar_Irradiance_MJ_m2"
    if target in corr.columns:
        top_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
        print(f"\n  Top correlates with Solar Irradiance:")
        for feat, val in top_corr.items():
            direction = "↑" if val > 0 else "↓"
            print(f"    {direction} {feat:<28} r = {val:+.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: SOLAR IRRADIANCE SEASONALITY (single year)
# ─────────────────────────────────────────────────────────────────────────────
def plot_solar_seasonality(df: pd.DataFrame):
    """
    Line plot of daily solar irradiance over one calendar year (2022).
    Overlays a 30-day rolling mean to highlight the seasonal trend clearly.
    Demonstrates the bimodal peak pattern typical of tropical Indian climates
    (pre-monsoon peak in April–May, post-monsoon secondary peak Oct–Nov).
    """
    print("\n  Generating Solar Seasonality Plot...")

    # Use year 2022 for the single-year view (middle of our 5-year dataset)
    year = 2022
    year_df = df[df.index.year == year].copy()

    if len(year_df) == 0:
        # Fallback to first available year
        year = df.index.year.min()
        year_df = df[df.index.year == year].copy()

    target = "Solar_Irradiance_MJ_m2"
    rolling_mean = year_df[target].rolling(window=30, center=True).mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Daily irradiance (faded)
    ax.fill_between(year_df.index, year_df[target], alpha=0.18,
                    color=PALETTE["primary"], label="Daily Irradiance")
    ax.plot(year_df.index, year_df[target], color=PALETTE["primary"],
            linewidth=0.9, alpha=0.5)

    # 30-day rolling mean
    ax.plot(year_df.index, rolling_mean, color=PALETTE["secondary"],
            linewidth=2.5, label="30-Day Rolling Mean", zorder=5)

    # Mark monsoon season (June–September)
    monsoon_start = pd.Timestamp(f"{year}-06-01")
    monsoon_end   = pd.Timestamp(f"{year}-09-30")
    ax.axvspan(monsoon_start, monsoon_end, alpha=0.08, color="#3B82F6",
               label="Monsoon Season (Jun–Sep)")

    ax.set_title(
        f"Daily Solar Irradiance Seasonality — Year {year}\n"
        f"Jabalpur, MP, India (23.18°N, 79.98°E)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Solar Irradiance (MJ/m²/day)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=0)

    output_path = os.path.join(OUTPUT_DIR, "eda_solar_seasonality.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✅ Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: FEATURE DISTRIBUTION HISTOGRAMS
# ─────────────────────────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame):
    """
    Histogram + KDE for every feature to assess data shape,
    detect outliers, and check normality assumptions.
    """
    print("\n  Generating Feature Distribution Plots...")

    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = axes.flatten()

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["danger"], "#8B5CF6", "#EC4899"]

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.set_facecolor(PALETTE["bg"])
        data = df[col].dropna()

        sns.histplot(data, ax=ax, kde=True, color=colors[i % len(colors)],
                     edgecolor="white", linewidth=0.4, alpha=0.75)

        ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=9)

        # Annotate mean ± std
        mu, sigma = data.mean(), data.std()
        ax.axvline(mu, color="black", linestyle="--", linewidth=1.2,
                   label=f"μ={mu:.1f}")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions — NASA POWER Meteorological Variables\n"
        "Jabalpur, MP, India | 2019–2023",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "eda_feature_distributions.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✅ Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: MONTHLY BOX PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_boxplot(df: pd.DataFrame):
    """
    Box plot of solar irradiance grouped by calendar month.
    Reveals the seasonal distribution, median values per month,
    and variance — critical context for a research paper.
    """
    print("\n  Generating Monthly Boxplot...")

    target = "Solar_Irradiance_MJ_m2"
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    df_box = df[[target]].copy()
    df_box["Month"] = df_box.index.month

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Fix: assign Month to hue to silence seaborn FutureWarning
    df_box["Month_str"] = df_box["Month"].astype(str)
    sns.boxplot(
        data       = df_box,
        x          = "Month",
        y          = target,
        hue        = "Month",
        palette    = "Blues",
        linewidth  = 1.2,
        flierprops = dict(marker="o", markersize=3, alpha=0.4),
        legend     = False,
        ax         = ax,
    )

    # Fix: use FixedLocator before set_xticklabels to silence UserWarning
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_title(
        "Monthly Distribution of Solar Irradiance (2019–2023)\n"
        "Jabalpur, Madhya Pradesh, India",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Solar Irradiance (MJ/m²/day)", fontsize=11)

    output_path = os.path.join(OUTPUT_DIR, "eda_solar_monthly_boxplot.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✅ Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SCRIPT 03: Exploratory Data Analysis")
    print("=" * 65)

    df = load_data()

    plot_correlation_heatmap(df)
    plot_solar_seasonality(df)
    plot_distributions(df)
    plot_monthly_boxplot(df)

    print("\n" + "=" * 65)
    print("  Script 03 complete. Run 04_feature_engineering.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
