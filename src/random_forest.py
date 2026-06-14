from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import os

# =====================================
# Load train-test split data
# =====================================

X_train = np.load("data/proccessed/X_train.npy")
X_test = np.load("data/proccessed/X_test.npy")

y_train = np.load("data/proccessed/y_train.npy").squeeze()
y_test = np.load("data/proccessed/y_test.npy").squeeze()

# =====================================
# Train Random Forest
# =====================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =====================================
# Make Predictions
# =====================================

pred_rf = rf.predict(X_test)

# =====================================
# Calculate Metrics
# =====================================

mae = mean_absolute_error(y_test, pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, pred_rf))
r2 = r2_score(y_test, pred_rf)

print("\n===== RANDOM FOREST RESULTS =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================
# Load Linear Regression Metrics
# =====================================

with open("reports/metrics_linear.json", "r") as f:
    linear_metrics = json.load(f)

linear_mae = linear_metrics["MAE"]
linear_rmse = linear_metrics["RMSE"]
linear_r2 = linear_metrics["R2"]

# =====================================
# Compare Models
# =====================================

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, mae],
    "RMSE": [linear_rmse, rmse],
    "R2": [linear_r2, r2]
})

print("\n===== MODEL COMPARISON =====")
print(comparison)

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# =====================================
# Feature Importance Plot
# =====================================

importances = rf.feature_importances_

features = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm"
]

plt.figure(figsize=(6, 4))
plt.bar(features, importances)

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()

os.makedirs("reports/figures", exist_ok=True)

plt.savefig(
    "reports/figures/rf_importance.png",
    dpi=150
)

plt.show()

print("\n===== FEATURE IMPORTANCE =====")

for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

# =====================================
# Save Model
# =====================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    rf,
    "models/random_forest.joblib"
)

print("\nModel saved successfully.")
print("Location: models/random_forest.joblib")