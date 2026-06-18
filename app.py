import streamlit as st
import pandas as pd

from src.predict import predict_yield

# Page settings
st.set_page_config(
    page_title="Mushroom Yield Forecast",
    layout="centered"
)

# Title
st.title("🍄 Polyhouse Yield Predictor")
st.write("Predict oyster mushroom yield using environmental sensor readings.")

# Sidebar
with st.sidebar:
    st.header("Sensor Inputs")

    temp = st.slider(
        "Temperature (°C)",
        min_value=10.0,
        max_value=35.0,
        value=22.0,
        step=0.1
    )

    humid = st.slider(
        "Humidity (%)",
        min_value=50.0,
        max_value=100.0,
        value=88.0,
        step=0.5
    )

    co2 = st.slider(
        "CO₂ (ppm)",
        min_value=400,
        max_value=2000,
        value=900,
        step=10
    )

# Warnings
if temp < 15 or temp > 30:
    st.warning("Temperature is outside the typical training range.")

if humid < 70 or humid > 98:
    st.warning("Humidity is outside the typical training range.")

if co2 < 500 or co2 > 1500:
    st.warning("CO₂ is outside the typical training range.")

# Prediction button
if st.button("Predict Yield"):

    prediction = predict_yield(
        temperature_c=temp,
        humidity_pct=humid,
        co2_ppm=co2
    )

    st.metric(
        label="Estimated Daily Yield",
        value=f"{prediction:.2f} kg"
    )

# Chart
st.subheader("Humidity Reference Trend")

chart_df = pd.DataFrame({
    "Humidity (%)": [70, 75, 80, 85, 90, 95, 98],
    "Relative Yield Trend": [70, 75, 82, 88, 94, 97, 100]
})

st.line_chart(
    chart_df.set_index("Humidity (%)")
)

# Model Information
with st.expander("Model Information"):
    st.write("Model: Random Forest Regressor")
    st.write("Version: v1.0")
    st.write("Target: Oyster Mushroom Yield")

# Footer
st.markdown("---")
st.caption("Mushroom Yield Forecasting | Internship Project")