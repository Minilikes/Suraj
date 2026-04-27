# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT 01: NASA POWER API – Data Fetching (Multi-City)
=============================================================================
Fetches 5 years of NASA POWER data (2019–2023) for all cities defined
in cities.py and saves a separate CSV per city in outputs/cities/
=============================================================================
"""

import requests
import pandas as pd
import os
from cities import CITIES

# ── CONFIGURATION ────────────────────────────────────────────────────────────
START_DATE = "20190101"
END_DATE   = "20231231"
OUTPUT_DIR = os.path.join("outputs", "cities")
BASE_URL   = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = ",".join([
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "CLOUD_AMT",
    "WS2M",
    "PRECTOTCORR",
])

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_nasa_power_data(lat, lon, start, end, params):
    query_params = {
        "parameters": params,
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start,
        "end":        end,
        "format":     "JSON",
    }
    response = requests.get(BASE_URL, params=query_params, timeout=120)
    response.raise_for_status()
    return response.json()

# ── PARSE ─────────────────────────────────────────────────────────────────────
def parse_to_dataframe(api_response):
    properties     = api_response.get("properties", {})
    parameter_data = properties.get("parameter", {})

    if not parameter_data:
        raise ValueError("Unexpected API response structure.")

    df = pd.DataFrame(parameter_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "Date"
    df = df.reset_index()

    df.rename(columns={
        "ALLSKY_SFC_SW_DWN": "Solar_Irradiance_MJ_m2",
        "T2M":               "Temperature_C",
        "RH2M":              "Relative_Humidity_pct",
        "CLOUD_AMT":         "Cloud_Fraction",
        "WS2M":              "Wind_Speed_ms",
        "PRECTOTCORR":       "Precipitation_mm",
    }, inplace=True)

    return df

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  S.U.R.A.J. — Multi-City NASA Data Fetch")
    print("=" * 65)

    for city_name, info in CITIES.items():
        print(f"\n  Fetching: {city_name} ({info['lat']}°N, {info['lon']}°E)...")

        try:
            api_data = fetch_nasa_power_data(
                lat    = info["lat"],
                lon    = info["lon"],
                start  = START_DATE,
                end    = END_DATE,
                params = PARAMETERS,
            )

            df = parse_to_dataframe(api_data)
            df.replace(-999.0, pd.NA, inplace=True)

            # Save as outputs/cities/jabalpur_raw.csv etc.
            filename = os.path.join(OUTPUT_DIR, f"{city_name.lower()}_raw.csv")
            df.to_csv(filename, index=False)

            print(f"  [OK] {city_name}: {len(df)} rows saved → {filename}")

        except Exception as e:
            print(f"  [ERROR] {city_name} failed: {e}")

    print("\n" + "=" * 65)
    print("  All cities fetched. Run 02_data_preprocessing.py next.")
    print("=" * 65)

if __name__ == "__main__":
    main()