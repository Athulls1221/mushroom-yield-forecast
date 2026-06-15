import numpy as np
import pandas as pd
import json
import time
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Data
# ==========================================

X_train = np.load("data/proccessed/X_train.npy")
X_test = np.load("data/proccessed/X_test.npy")

y_train = np.load("data/proccessed/y_train.npy").squeeze()
y_test = np.load("data/proccessed/y_test.npy").squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# ==========================================
# Time Series Split
# ==========================================

tscv = TimeSeriesSplit(n_splits=3)

# ==========================================
# Parameter Grid
# ==========================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 5]
}

# ==========================================
# Model
# ==========================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ==========================================
# Grid Search
# ==========================================

start = time.time()

search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

search.fit(X_train, y_train)

runtime = time.time() - start

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest CV MAE:")
print(-search.best_score_)

# ==========================================
# Save CV Results
# ==========================================

cv_results = pd.DataFrame(search.cv_results_)

cv_results.to_csv(
    "reports/search_cv_results.csv",
    index=False
)

# ==========================================
# Save Best Parameters
# ==========================================

with open(
    "models/rf_best_params.json",
    "w"
) as f:
    json.dump(
        search.best_params_,
        f,
        indent=4
    )

# ==========================================
# Save Best Model
# ==========================================

best_model = search.best_estimator_

joblib.dump(
    best_model,
    "models/rf_best_model.pkl"
)

# ==========================================
# Test Evaluation
# ==========================================

preds = best_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        preds
    )
)

r2 = r2_score(
    y_test,
    preds
)

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "runtime_seconds": float(runtime)
}

with open(
    "reports/tuned_rf_metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

# ==========================================
# Final Output
# ==========================================

print("\n===== TUNED RANDOM FOREST RESULTS =====")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))
print("Runtime:", round(runtime, 2), "seconds")

print("\nSaved:")
print("models/rf_best_model.pkl")
print("models/rf_best_params.json")
print("reports/search_cv_results.csv")
print("reports/tuned_rf_metrics.json")