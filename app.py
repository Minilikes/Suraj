# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)

CITIES_DIR = os.path.join("outputs", "cities")

print("Loading city models...")
MODELS = {}
for city in ["Jabalpur", "Bhopal", "Delhi", "Mumbai", "Jaipur", "Ladakh"]:
    model_path = os.path.join(CITIES_DIR, f"{city.lower()}_model.joblib")
    if os.path.exists(model_path):
        MODELS[city] = joblib.load(model_path)
        print(f"  [OK] {city} model loaded")
print(f"\nAll models ready. Open http://127.0.0.1:5000 in your browser\n")


@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json()
    city      = data.get("city", "Jabalpur")
    cloud     = float(data.get("cloud", 50))
    temp      = float(data.get("temp", 25))
    humidity  = float(data.get("humidity", 60))
    yesterday = float(data.get("yesterday", 5.0))

    if city not in MODELS:
        return jsonify({"error": f"Model for {city} not found"}), 400

    from datetime import datetime
    today  = datetime.now()
    month  = today.month
    doy    = today.timetuple().tm_yday

    season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2,9:2, 10:3,11:3}
    season    = season_map[month]
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    doy_sin   = np.sin(2 * np.pi * doy / 365)
    doy_cos   = np.cos(2 * np.pi * doy / 365)

    features = np.array([[
        temp, humidity, cloud, 2.0, 0.0,
        month, doy, season,
        month_sin, month_cos, doy_sin, doy_cos,
        yesterday, yesterday, yesterday, yesterday, yesterday,
    ]])

    prediction = float(MODELS[city].predict(features)[0])
    prediction = round(max(0.0, prediction), 3)
    return jsonify({"prediction": prediction, "city": city})


@app.route("/status")
def status():
    return jsonify({"status": "S.U.R.A.J. API running", "cities": list(MODELS.keys())})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
