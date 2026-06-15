import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load latest feature dataset
# ==========================================

df = pd.read_parquet("data/proccessed/features.parquet")

print("Columns:")
print(df.columns.tolist())

# ==========================================
# Features and target
# ==========================================

X = df.drop(columns=["yield_kg"])
y = df["yield_kg"]

print("\nFeature count:", X.shape[1])

# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ==========================================
# Train Model
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "models/linear_regression.joblib"
)

# ==========================================
# Evaluation
# ==========================================

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\n===== LINEAR REGRESSION RESULTS =====")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))

print("\nModel saved:")
print("models/linear_regression.joblib")

print("\nModel expects:")
print(model.n_features_in_, "features")