\# Mushroom Yield Forecast



\## Problem Statement



This project aims to predict mushroom yield using polyhouse sensor data such as temperature, humidity, and CO₂ levels.



The objective is to build a machine learning pipeline that helps farmers monitor growing conditions and forecast mushroom production.



\## Project Structure



mushroom-yield-forecast/



├── data/

│   └── raw/



├── models/



├── notebooks/



├── src/

│   └── smoke\_test.py



├── app.py



├── requirements.txt



├── README.md



└── .gitignore



\## Environment Setup



Activate the virtual environment:



```powershell

\& ..\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



\## Run Smoke Test



```powershell

python .\\src\\smoke\_test.py

```



Expected output:



```text

Environment OK.

```



\## Git Workflow



Commit changes using descriptive commit messages:



```powershell

git add .

git commit -m "feat: add new functionality"

git push

```
----------------------------------------------------------------------------------------------

## Day 7 Feature Engineering

### Features Used

| Feature                | Formula                            |
| ---------------------- | ---------------------------------- |
| temperature_c          | Raw temperature value              |
| humidity_pct           | Raw humidity value                 |
| co2_ppm                | Raw CO₂ concentration              |
| temp_humid_interaction | temperature_c × humidity_pct ÷ 100 |

### Scaling

MinMaxScaler was applied to all feature columns:

scaled_value = (x - min) / (max - min)

The fitted scaler is saved as:

models/minmax_scaler.joblib

Note: For learning purposes, the scaler was fitted on the full cleaned dataset. In a production pipeline, scaling should be fitted only on training data to prevent data leakage.


