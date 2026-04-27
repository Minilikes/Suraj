# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from cities import CITIES

CITIES_DIR = os.path.join("outputs", "cities")
TARGET_COL = "Solar_Irradiance_MJ_m2"

def add_features(df):
    df["Month"]       = df.index.month
    df["Day_of_Year"] = df.index.dayofyear

    df["Season"] = df["Month"].map({
        12:0, 1:0, 2:0,
        3:1,  4:1, 5:1,
        6:2,  7:2, 8:2, 9:2,
        10:3, 11:3
    })

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOY_sin"]   = np.sin(2 * np.pi * df["Day_of_Year"] / 365)
    df["DOY_cos"]   = np.cos(2 * np.pi * df["Day_of_Year"] / 365)

    df["Solar_Lag_1"]  = df[TARGET_COL].shift(1)
    df["Solar_Lag_2"]  = df[TARGET_COL].shift(2)
    df["Solar_Lag_7"]  = df[TARGET_COL].shift(7)
    df["Solar_Roll7"]  = df[TARGET_COL].shift(1).rolling(7).mean()
    df["Solar_Roll30"] = df[TARGET_COL].shift(1).rolling(30).mean()

    df.dropna(inplace=True)
    return df

def main():
    print("=" * 65)
    print("  S.U.R.A.J. — Multi-City Feature Engineering")
    print("=" * 65)

    for city_name in CITIES:
        print(f"\n  Engineering features: {city_name}...")
        input_csv  = os.path.join(CITIES_DIR, f"{city_name.lower()}_cleaned.csv")
        output_csv = os.path.join(CITIES_DIR, f"{city_name.lower()}_features.csv")

        try:
            df = pd.read_csv(input_csv, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df = add_features(df)
            df.to_csv(output_csv, index=True, index_label="Date")
            print(f"  [OK] {len(df)} rows, {len(df.columns)} columns -> {output_csv}")
        except Exception as e:
            print(f"  [ERROR] {city_name}: {e}")

    print("\n" + "=" * 65)
    print("  Feature engineering complete. Run model training next.")
    print("=" * 65)

if __name__ == "__main__":
    main()
