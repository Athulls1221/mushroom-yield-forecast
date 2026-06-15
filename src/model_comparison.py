import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load Test Data
# =====================================

X_test = np.load("data/proccessed/X_test.npy")
y_test = np.load("data/proccessed/y_test.npy").squeeze()

# =====================================
# Load Models
# =====================================

linear_model = joblib.load(
    "models/linear_regression.joblib"
)

rf_model = joblib.load(
    "models/random_forest.joblib"
)

tuned_rf_model = joblib.load(
    "models/rf_best_model.pkl"
)

# =====================================
# Make Predictions
# =====================================

linear_pred = linear_model.predict(X_test)

rf_pred = rf_model.predict(X_test)

tuned_rf_pred = tuned_rf_model.predict(X_test)

# =====================================
# Calculate Metrics
# =====================================

results = []

for name, pred in [
    ("Linear Regression", linear_pred),
    ("Random Forest", rf_pred),
    ("Tuned Random Forest", tuned_rf_pred)
]:
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append([
        name,
        round(mae, 4),
        round(rmse, 4),
        round(r2, 4)
    ])

comparison = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
)

print("\n===== MODEL COMPARISON =====")
print(comparison)

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# =====================================
# Actual vs Predicted Plot
# =====================================

plt.figure(figsize=(6, 4))

plt.scatter(
    y_test,
    linear_pred,
    alpha=0.6,
    label="Linear Regression"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/pred_vs_actual.png",
    dpi=150
)

plt.show()

print("\nPlot saved to reports/pred_vs_actual.png")