import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_parquet("data/interim/02_cleaned.parquet").sort_values("timestamp")
feature_cols = ["temperature_c", "humidity_pct", "co2_ppm"]

split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])
y_train = train["yield_kg"].values
y_test = test["yield_kg"].values

joblib.dump(scaler, "models/minmax_scaler_train.joblib")

import numpy as np

np.save("data/proccessed/X_train.npy", X_train)
np.save("data/proccessed/X_test.npy", X_test)
np.save("data/proccessed/y_train.npy", y_train)
np.save("data/proccessed/y_test.npy", y_test)

print(f"Train: {train['timestamp'].min()} → {train['timestamp'].max()}")
print(f"Test:  {test['timestamp'].min()} → {test['timestamp'].max()}")