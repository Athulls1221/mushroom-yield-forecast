from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

"""
Day 7: Feature Engineering & Min-Max Scaling

Feature definitions:
- temperature_c: Polyhouse temperature in Celsius.
- humidity_pct: Relative humidity percentage.
- co2_ppm: Carbon dioxide concentration in ppm.
- temp_humid_interaction = temperature_c * humidity_pct / 100

Note:
For today, scaler is fitted on the full cleaned dataset only to understand
MinMaxScaler.

On Day 8, scaling should be fitted only on training data to avoid data leakage.
"""

# Paths
DATA_PATH = Path("data/proccessed/02_cleaned.parquet")
FEATURE_PATH = Path("data/proccessed/features.parquet")
SCALER_PATH = Path("models/minmax_scaler.joblib")

# Create directories if they don't exist
Path("data/proccessed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_parquet(DATA_PATH).sort_values("timestamp")

# Feature Engineering
df["temp_humid_interaction"] = (
    df["temperature_c"] * df["humidity_pct"] / 100
)

# Feature columns
feature_cols = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "temp_humid_interaction"
]

X = df[feature_cols]
y = df["yield_kg"]

# Data validation
if X.isna().sum().sum() > 0:
    raise ValueError("NaNs found in feature matrix X")

if y.isna().sum() > 0:
    raise ValueError("NaNs found in target y")

# Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, SCALER_PATH)

# Create scaled feature dataframe
processed = pd.DataFrame(
    X_scaled,
    columns=[f"{col}_scaled" for col in feature_cols]
)

processed["yield_kg"] = y.values

# Save features
processed.to_parquet(FEATURE_PATH, index=False)

# Output summary
print("Feature engineering complete.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nScaled feature min values:")
print(processed.drop(columns=["yield_kg"]).min())

print("\nScaled feature max values:")
print(processed.drop(columns=["yield_kg"]).max())

print(f"\nSaved features -> {FEATURE_PATH}")
print(f"Saved scaler -> {SCALER_PATH}")