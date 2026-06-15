import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

# Load artifacts
_scaler = joblib.load(MODEL_DIR / "minmax_scaler_train.joblib")
_model = joblib.load(MODEL_DIR / "random_forest.joblib")


def predict_yield(
    temperature_c: float,
    humidity_pct: float,
    co2_ppm: float
) -> float:

    # Original features
    raw = pd.DataFrame({
        "temperature_c": [temperature_c],
        "humidity_pct": [humidity_pct],
        "co2_ppm": [co2_ppm]
    })

    # Scale the 3 features
    scaled = _scaler.transform(raw)

    # Interaction feature
    interaction = temperature_c * humidity_pct

    # Create final 4-feature input
    final_features = np.column_stack([
        scaled,
        [interaction]
    ])

    prediction = _model.predict(final_features)[0]

    return float(prediction)


if __name__ == "__main__":
    pred = predict_yield(
        temperature_c=22.0,
        humidity_pct=88.0,
        co2_ppm=920.0
    )

    print(f"Predicted Yield: {pred:.2f} kg")