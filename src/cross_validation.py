from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import os

# =====================================
# Load TRAIN data only
# =====================================

X_train = np.load("data/proccessed/X_train.npy")
y_train = np.load("data/proccessed/y_train.npy").squeeze()

# =====================================
# Time Series Split
# =====================================

tscv = TimeSeriesSplit(n_splits=5)

# =====================================
# Models
# =====================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

lin = LinearRegression()

# =====================================
# Cross Validation
# =====================================

rf_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

lin_scores = cross_val_score(
    lin,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

# Convert negative MAE to positive

rf_mae = -rf_scores
lin_mae = -lin_scores

print("\n===== CROSS VALIDATION RESULTS =====")

print("\nRandom Forest")
print(f"Mean CV MAE : {rf_mae.mean():.4f}")
print(f"Std CV MAE  : {rf_mae.std():.4f}")

print("\nLinear Regression")
print(f"Mean CV MAE : {lin_mae.mean():.4f}")
print(f"Std CV MAE  : {lin_mae.std():.4f}")

# =====================================
# Save Results
# =====================================

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "CV_MAE": [
        lin_mae.mean(),
        rf_mae.mean()
    ],
    "CV_STD": [
        lin_mae.std(),
        rf_mae.std()
    ]
})

os.makedirs("reports", exist_ok=True)

results.to_csv(
    "reports/cv_results.csv",
    index=False
)

print("\nResults saved to reports/cv_results.csv")