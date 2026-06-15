import json
from pathlib import Path

feature_cols = [
    "temperature_c_scaled",
    "humidity_pct_scaled",
    "co2_ppm_scaled",
    "temp_humid_interaction_scaled"
]

Path("models").mkdir(exist_ok=True)

with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=4)

print("feature_cols.json saved")