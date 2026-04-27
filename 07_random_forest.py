# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from cities import CITIES
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

CITIES_DIR = os.path.join("outputs", "cities")
TARGET_COL = "Solar_Irradiance_MJ_m2"

FEATURES = [
    "Temperature_C", "Relative_Humidity_pct", "Cloud_Fraction",
    "Wind_Speed_ms", "Precipitation_mm", "Month", "Day_of_Year",
    "Season", "Month_sin", "Month_cos", "DOY_sin", "DOY_cos",
    "Solar_Lag_1", "Solar_Lag_2", "Solar_Lag_7",
    "Solar_Roll7", "Solar_Roll30"
]

def train_city(city_name):
    input_csv  = os.path.join(CITIES_DIR, f"{city_name.lower()}_features.csv")
    model_path = os.path.join(CITIES_DIR, f"{city_name.lower()}_model.joblib")

    df = pd.read_csv(input_csv, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    X = df[FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"  [OK] {city_name}: R²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  -> {model_path}")

def main():
    print("=" * 65)
    print("  S.U.R.A.J. — Multi-City Model Training")
    print("=" * 65)

    for city_name in CITIES:
        print(f"\n  Training: {city_name}...")
        try:
            train_city(city_name)
        except Exception as e:
            print(f"  [ERROR] {city_name}: {e}")

    print("\n" + "=" * 65)
    print("  All models trained. Run dashboard update next.")
    print("=" * 65)

if __name__ == "__main__":
    main()
