app_code = """# -*- coding: utf-8 -*-
\"\"\"
=============================================================================
S.U.R.A.J. — Flask Backend API
=============================================================================
Loads all 6 city models and serves predictions to the dashboard.
Run this file, then open dashboard/index.html in your browser.
=============================================================================
\"\"\"

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow the dashboard HTML file to call this API

CITIES_DIR = os.path.join("outputs", "cities")

# Load all 6 models into memory at startup
print("Loading city models...")
MODELS = {}
for city in ["Jabalpur", "Bhopal", "Delhi", "Mumbai", "Jaipur", "Ladakh"]:
    model_path = os.path.join(CITIES_DIR, f"{city.lower()}_model.joblib")
    if os.path.exists(model_path):
        MODELS[city] = joblib.load(model_path)
        print(f"  [OK] {city} model loaded")
    else:
        print(f"  [MISSING] {city} model not found at {model_path}")

print(f"\\nAll models ready. Starting server...\\n")


@app.route("/predict", methods=["POST"])
def predict():
    \"\"\"
    Accepts JSON: { city, cloud, temp, humidity, yesterday }
    Returns JSON: { prediction, city }
    \"\"\"
    data = request.get_json()

    city      = data.get("city", "Jabalpur")
    cloud     = float(data.get("cloud", 50))
    temp      = float(data.get("temp", 25))
    humidity  = float(data.get("humidity", 60))
    yesterday = float(data.get("yesterday", 5.0))

    if city not in MODELS:
        return jsonify({"error": f"Model for {city} not found"}), 400

    model = MODELS[city]

    # Build the feature vector matching training order
    # Features: Temperature_C, Relative_Humidity_pct, Cloud_Fraction,
    #           Wind_Speed_ms, Precipitation_mm, Month, Day_of_Year,
    #           Season, Month_sin, Month_cos, DOY_sin, DOY_cos,
    #           Solar_Lag_1, Solar_Lag_2, Solar_Lag_7, Solar_Roll7, Solar_Roll30

    from datetime import datetime
    today     = datetime.now()
    month     = today.month
    doy       = today.timetuple().tm_yday

    season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2,9:2, 10:3,11:3}
    season    = season_map[month]

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    doy_sin   = np.sin(2 * np.pi * doy / 365)
    doy_cos   = np.cos(2 * np.pi * doy / 365)

    # Cloud fraction from dashboard is 0-100, model trained on 0-100 scale
    features = np.array([[
        temp,        # Temperature_C
        humidity,    # Relative_Humidity_pct
        cloud,       # Cloud_Fraction
        2.0,         # Wind_Speed_ms (typical average)
        0.0,         # Precipitation_mm (assume dry)
        month,       # Month
        doy,         # Day_of_Year
        season,      # Season
        month_sin,   # Month_sin
        month_cos,   # Month_cos
        doy_sin,     # DOY_sin
        doy_cos,     # DOY_cos
        yesterday,   # Solar_Lag_1
        yesterday,   # Solar_Lag_2 (use same as approximation)
        yesterday,   # Solar_Lag_7 (use same as approximation)
        yesterday,   # Solar_Roll7
        yesterday,   # Solar_Roll30
    ]])

    prediction = float(model.predict(features)[0])
    prediction = round(max(0.0, prediction), 3)

    return jsonify({"prediction": prediction, "city": city})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "S.U.R.A.J. API running", "cities": list(MODELS.keys())})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)
print("app.py written OK")
