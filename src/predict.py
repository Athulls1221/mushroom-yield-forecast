import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "minmax_scaler_train.joblib")
    model = joblib.load(MODEL_DIR / "random_forest.joblib")
    return scaler, model


_scaler, _model = load_artifacts()


def predict_yield(
    temperature_c: float,
    humidity_pct: float,
    co2_ppm: float
) -> float:

    # Create input dataframe
    raw = pd.DataFrame({
        "temperature_c": [temperature_c],
        "humidity_pct": [humidity_pct],
        "co2_ppm": [co2_ppm]
    })

    # Scale original features
    scaled = _scaler.transform(raw)

    # Interaction feature (same as training)
    interaction = temperature_c * humidity_pct

    # Final feature set
    final_features = np.column_stack([
        scaled,
        [interaction]
    ])

    # Predict
    prediction = _model.predict(final_features)[0]

    return float(prediction)


if __name__ == "__main__":
    pred = predict_yield(
        temperature_c=22.0,
        humidity_pct=88.0,
        co2_ppm=920.0
    )

    print(f"Predicted Yield: {pred:.2f} kg")