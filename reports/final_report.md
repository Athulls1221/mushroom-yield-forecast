\# Mushroom Yield Forecast – Technical Report



\## Executive Summary



This project predicts oyster mushroom yield using environmental sensor data collected from a polyhouse environment. A Random Forest Regression model was trained using temperature, humidity, and CO₂ sensor readings. The trained model was deployed through a Streamlit web application and later hosted on Streamlit Community Cloud.



\---



\## 1. Problem Statement



Mushroom cultivation is highly dependent on environmental conditions. Small changes in temperature, humidity, and CO₂ concentration can affect daily yield.



The objective of this project is to develop a machine learning model capable of predicting oyster mushroom yield from sensor readings and provide predictions through an easy-to-use web application.



\---



\## 2. Dataset Description



The dataset contains:



\- Temperature (°C)

\- Humidity (%)

\- CO₂ (ppm)

\- Yield (kg)



Data preprocessing included:



\- Missing value handling

\- Duplicate removal

\- Feature scaling

\- Feature engineering



\---



\## 3. Exploratory Data Analysis



Exploratory analysis was performed to understand:



\- Data distributions

\- Feature correlations

\- Sensor trends

\- Yield relationships



Visualizations were generated using matplotlib and pandas.



\---



\## 4. Feature Engineering



An interaction feature was created:



Temperature × Humidity



This feature helps the model capture combined environmental effects on mushroom growth.



Features used:



\- Temperature

\- Humidity

\- CO₂

\- Temperature × Humidity



\---



\## 5. Model Development



Several machine learning approaches were evaluated.



Final selected model:



\*\*Random Forest Regressor\*\*



Reasons for selection:



\- Handles non-linear relationships

\- Robust to noise

\- Good prediction accuracy

\- Minimal preprocessing requirements



\---



\## 6. Model Evaluation



Evaluation metrics:



\- MAE (Mean Absolute Error)

\- RMSE (Root Mean Squared Error)

\- R² Score



The model achieved acceptable performance on the test dataset and was selected as the champion model for deployment.



\---



\## 7. Streamlit Application



A Streamlit application was developed to provide predictions through an interactive user interface.



Features:



\- Temperature slider

\- Humidity slider

\- CO₂ slider

\- Yield prediction button

\- Model information panel

\- Humidity trend chart

\- Input validation warnings



\---



\## 8. Cloud Deployment



The application was deployed using Streamlit Community Cloud.



Deployment URL:



https://mushroom-yield-forecast-apjbqtsnkmpwcwqbbbzmcz.streamlit.app



Repository:



https://github.com/Athulls1221/mushroom-yield-forecast



\---



\## 9. Monitoring and Logging



Prediction logging was implemented using CSV files.



Logged fields:



\- Timestamp

\- Temperature

\- Humidity

\- CO₂

\- Predicted Yield



Monitoring focuses on:



\- Prediction trends

\- Data drift

\- Sensor anomalies

\- Model performance degradation



\---



\## 10. Future Improvements



Planned enhancements:



1\. Real-time IoT sensor integration

2\. Larger training dataset

3\. Automated model retraining

4\. Farmer dashboard analytics

5\. Mobile-friendly interface



\---



\## 11. Limitations



Current limitations include:



\- Limited dataset size

\- Synthetic training data

\- Limited environmental variables

\- No real-time sensor integration



Future versions will address these limitations.



\---



\## Reproduction Steps



Clone repository:



```bash

git clone https://github.com/Athulls1221/mushroom-yield-forecast.git

```



Create virtual environment:



```bash

python -m venv venv

```



Activate virtual environment:



```bash

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run application:



```bash

streamlit run app.py

```



Run tests:



```bash

pytest tests/

```



\---



\## Conclusion



This project successfully demonstrates an end-to-end machine learning workflow including data preprocessing, feature engineering, model training, evaluation, deployment, monitoring, and documentation. The deployed Streamlit application provides an accessible interface for predicting oyster mushroom yield using environmental sensor readings.

