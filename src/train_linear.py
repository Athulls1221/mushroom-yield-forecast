import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import json
import os

# Load train-test split data
X_train = np.load("data/proccessed/X_train.npy")
X_test =np.load("data/proccessed/X_test.npy")

y_train = np.load("data/proccessed/y_train.npy").squeeze()
y_test = np.load("data/proccessed/y_test.npy").squeeze()

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred_test = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

print(f"Test MAE:  {mae:.2f} kg")
print(f"Test RMSE: {rmse:.2f} kg")
print(f"Test R²:   {r2:.3f}")

print("\nCoefficients:")
for name, coef in zip(["temp", "humidity", "co2"], model.coef_):
    print(f"  coef {name}: {coef:.3f}")

# Save metrics
os.makedirs("reports", exist_ok=True)

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open("reports/metrics_linear.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_regression.joblib")

print("\nMetrics saved to reports/metrics_linear.json")
print("Model saved to models/linear_regression.joblib")