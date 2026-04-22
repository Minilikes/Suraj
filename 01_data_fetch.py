# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 01: NASA POWER API — Data Fetching
=============================================================================
Research Project : Predictive Solar Energy Forecasting for Tropical India
Target Location  : Jabalpur, Madhya Pradesh (23.18°N, 79.98°E)
Data Period      : 2019-01-01 to 2023-12-31 (5 Years)
API              : NASA POWER Daily Climatology API v2
Output           : outputs/nasa_power_raw.csv

Parameters Fetched:
  ALLSKY_SFC_SW_DWN  → All-Sky Surface Shortwave Downward Irradiance (MJ/m²/day)
  T2M                → Temperature at 2 Meters (°C)
  RH2M               → Relative Humidity at 2 Meters (%)
  CLDFRC             → Cloud Amount (fraction, 0–1)
  WS2M               → Wind Speed at 2 Meters (m/s)
  PRECTOTCORR        → Precipitation (mm/day)
=============================================================================
"""

import requests
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LATITUDE   = 23.18       # Jabalpur, Madhya Pradesh
LONGITUDE  = 79.98
START_DATE = "20190101"
END_DATE   = "20231231"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "nasa_power_raw.csv")

# NASA POWER API v2 — Daily Climatology Endpoint
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Meteorological parameters to fetch
PARAMETERS = ",".join([
    "ALLSKY_SFC_SW_DWN",   # Solar Irradiance (target variable)
    "T2M",                  # Temperature at 2m
    "RH2M",                 # Relative Humidity at 2m
    "CLOUD_AMT",            # Cloud Amount (Fraction)
    "WS2M",                 # Wind Speed
    "PRECTOTCORR",          # Precipitation
])

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# API REQUEST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def fetch_nasa_power_data(lat: float, lon: float, start: str, end: str,
                           params: str) -> dict:
    """
    Fetches daily meteorological data from NASA POWER API.

    Args:
        lat   : Latitude (decimal degrees)
        lon   : Longitude (decimal degrees)
        start : Start date in YYYYMMDD format
        end   : End date in YYYYMMDD format
        params: Comma-separated parameter string

    Returns:
        JSON response dictionary from NASA POWER API
    """
    query_params = {
        "parameters"  : params,
        "community"   : "RE",           # Renewable Energy community dataset
        "longitude"   : lon,
        "latitude"    : lat,
        "start"       : start,
        "end"         : end,
        "format"      : "JSON",
    }

    print("=" * 65)
    print("  NASA POWER API — Data Fetch Request")
    print("=" * 65)
    print(f"  Location  : ({lat}°N, {lon}°E) — Jabalpur, MP, India")
    print(f"  Period    : {start} to {end}")
    print(f"  Parameters: {params.replace(',', ', ')}")
    print("-" * 65)
    print("  Sending API request... (this may take 10-30 seconds)")

    try:
        response = requests.get(BASE_URL, params=query_params, timeout=120)
        response.raise_for_status()
        print(f"  [OK] API Response Status: {response.status_code} OK")
        return response.json()

    except requests.exceptions.ConnectionError:
        print("  [ERROR] No internet connection or NASA server unreachable.")
        raise
    except requests.exceptions.Timeout:
        print("  [ERROR] Request timed out. Try again in a few minutes.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"  [HTTP ERROR] {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PARSE JSON → DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
def parse_to_dataframe(api_response: dict) -> pd.DataFrame:
    """
    Parses the NASA POWER JSON response into a clean Pandas DataFrame.

    The API returns data in a nested dict: properties → parameter → {YYYYDDD: value}
    We pivot this into a wide-format DataFrame with columns per parameter.

    Args:
        api_response: Raw JSON dict from NASA POWER API

    Returns:
        pd.DataFrame with DatetimeIndex and one column per parameter
    """
    print("\n  Parsing API response into DataFrame...")

    # Navigate nested JSON structure
    properties = api_response.get("properties", {})
    parameter_data = properties.get("parameter", {})

    if not parameter_data:
        raise ValueError("Unexpected API response structure. 'parameter' key missing.")

    # Build DataFrame: each parameter is a column, keys are date strings (YYYYMMDD)
    df = pd.DataFrame(parameter_data)

    # The index contains date strings in YYYYMMDD format — convert to datetime
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "Date"

    # Reset to get Date as a regular column (easier for CSV readability)
    df = df.reset_index()

    # Rename columns to descriptive names for research clarity
    column_rename = {
        "ALLSKY_SFC_SW_DWN" : "Solar_Irradiance_MJ_m2",
        "T2M"               : "Temperature_C",
        "RH2M"              : "Relative_Humidity_pct",
        "CLOUD_AMT"         : "Cloud_Fraction",
        "WS2M"              : "Wind_Speed_ms",
        "PRECTOTCORR"       : "Precipitation_mm",
    }
    df.rename(columns=column_rename, inplace=True)

    print(f"  [OK] DataFrame created: {len(df)} rows x {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Step 1: Fetch data from NASA POWER API
    api_data = fetch_nasa_power_data(
        lat    = LATITUDE,
        lon    = LONGITUDE,
        start  = START_DATE,
        end    = END_DATE,
        params = PARAMETERS,
    )

    # Step 2: Parse JSON → DataFrame
    df = parse_to_dataframe(api_data)

    # Step 3: Replace NASA fill/error values (-999) with NaN
    #         NASA POWER uses -999 as a sentinel for missing/invalid data
    FILL_VALUE = -999.0
    df.replace(FILL_VALUE, pd.NA, inplace=True)
    missing_count = df.isnull().sum().sum()

    print(f"  NASA fill values (-999) replaced with NaN.")
    print(f"  Total missing values detected: {missing_count}")

    # Step 4: Display preview
    print("\n" + "=" * 65)
    print("  DATASET PREVIEW (first 5 rows)")
    print("=" * 65)
    print(df.head().to_string(index=False))

    # Step 5: Dataset info
    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print("=" * 65)
    print(f"  Shape         : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Date Range    : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Missing Values: {missing_count}")
    print("\n  Descriptive Statistics:")
    print(df.describe().round(3).to_string())

    # Step 6: Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  [SAVED] Raw data saved -> {OUTPUT_CSV}")
    print("=" * 65)
    print("  Script 01 complete. Run 02_data_preprocessing.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
