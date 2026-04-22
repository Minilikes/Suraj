# ☀️ Predictive Solar Energy Forecasting — Research Project
### Localized ML-Based Forecasting for Tropical Indian Climate (23.18°N, 79.98°E)

---

## 📖 Project Overview

This project implements a **machine-learning-based solar irradiance forecasting pipeline** designed for publication in **Google Scholar / IEEE / Springer** journals. It uses real meteorological data from the **NASA POWER API** to predict daily solar irradiance for a specific tropical Indian location (Jabalpur, Madhya Pradesh).

**Research Focus:** Demonstrating the superiority of ensemble ML models (Random Forest, XGBoost) over naive persistence baselines for localized solar energy prediction in developing-nation smart grid contexts.

---

## 📁 Project Structure

```
suraj/
├── 01_data_fetch.py          # Step 1: Fetch 5 years of NASA POWER data
├── 02_data_preprocessing.py  # Step 2: Clean, handle nulls, set DatetimeIndex
├── 03_eda.py                 # Step 3: Exploratory Data Analysis (EDA)
├── 04_feature_engineering.py # Step 4: Time features + Lag features
├── 05_train_test_split.py    # Step 5: Temporal train/test split
├── 06_baseline_model.py      # Step 6: Persistence (naive) baseline model
├── 07_random_forest.py       # Step 7: Train Random Forest Regressor
├── 08_evaluation.py          # Step 8: Evaluate model (MAE, RMSE, R²)
├── 09_hyperparameter_tuning.py # Step 9: RandomizedSearchCV optimization
├── 10_feature_importance.py  # Step 10: Feature importance visualization
├── 11_actual_vs_predicted.py # Step 11: Publication-quality forecast plot
├── outputs/                  # Generated CSVs, plots, and model files
│   └── (auto-created)
└── README.md
```

---

## ⚙️ Installation

```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn xgboost joblib scipy
```

---

## 🚀 Run Order

Execute scripts **in order** (each depends on the previous):

```bash
python 01_data_fetch.py
python 02_data_preprocessing.py
python 03_eda.py
python 04_feature_engineering.py
python 05_train_test_split.py
python 06_baseline_model.py
python 07_random_forest.py
python 08_evaluation.py
python 09_hyperparameter_tuning.py
python 10_feature_importance.py
python 11_actual_vs_predicted.py
```

---

## 📍 Location Details

| Parameter | Value |
|---|---|
| Latitude | 23.18° N |
| Longitude | 79.98° E |
| Region | Jabalpur, Madhya Pradesh, India |
| Climate Type | Tropical / Sub-humid |
| Data Source | NASA POWER Daily Climatology API |
| Period | 2019–2023 (5 Years) |
