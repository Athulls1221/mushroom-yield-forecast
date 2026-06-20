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


## Train/Test Split & Data Leakage Prevention

### Overview

A chronological train/test split was performed to preserve the temporal nature of the mushroom yield dataset. This prevents future observations from influencing model training and provides a realistic evaluation of forecasting performance.

### Method

* Dataset sorted by `timestamp`
* Split ratio: 80% Training, 20% Testing
* Features:

  * temperature_c
  * humidity_pct
  * co2_ppm
* Target:

  * yield_kg

### Scaling

Min-Max scaling was applied to the feature variables.

To prevent data leakage:

* The scaler was fitted only on the training dataset.
* The same scaler was then used to transform the test dataset.
* The trained scaler was saved as:

`models/minmax_scaler_train.joblib`

### Saved Artifacts

Training and testing datasets were saved for downstream model training:

* `data/proccessed/X_train.npy`
* `data/proccessed/X_test.npy`
* `data/proccessed/y_train.npy`
* `data/proccessed/y_test.npy`

### Train/Test Date Ranges

Training Window:

* Start: [Replace with output from script]
* End: [Replace with output from script]

Testing Window:

* Start: [Replace with output from script]
* End: [Replace with output from script]

### Data Leakage Prevention

The following practices were used to prevent data leakage:

1. Chronological splitting was performed before model training.
2. Future observations were never used during training.
3. The scaler was fitted only on training data.
4. The test set remained isolated for final model evaluation.

## Day 9 – Baseline Linear Regression Model

### Overview

A baseline Linear Regression model was trained to predict mushroom yield using scaled environmental features.

### Input Features

* temperature_c
* humidity_pct
* co2_ppm

### Target Variable

* yield_kg

### Model

The model was trained using Scikit-Learn's `LinearRegression`.

### Evaluation Metrics

| Metric | Value   |
| ------ | ------- |
| MAE    | 0.42 kg |
| RMSE   | 0.54 kg |
| R²     | 0.427   |

### Interpretation

The baseline model explains approximately 42.7% of the variation in mushroom yield. While the model captures meaningful relationships between environmental conditions and yield, a significant portion of variability remains unexplained. This suggests that additional features or nonlinear relationships may improve performance.

### Saved Artifacts

* `models/linear_regression.joblib`
* `reports/metrics_linear.json`

---

## Day 10 – Residual Analysis & Model Diagnostics

### Objective

Residual analysis was performed to evaluate model performance beyond aggregate metrics and identify systematic prediction errors.

Residuals were calculated as:

Residual = Actual Yield − Predicted Yield

### Diagnostic Figures

Generated figures:

* `reports/figures/residuals_linear.png`
* `reports/figures/residuals_vs_humidity.png`

### Findings

1. Residuals are generally centered around zero.
2. The model captures part of the yield signal but leaves substantial unexplained variation.
3. No severe prediction bias is observed, although prediction errors increase in some regions of the feature space.
4. The residual spread suggests that linear regression may not fully model the underlying relationships.

### Recommendation

Linear Regression provides a useful baseline model. However, the moderate R² score and residual patterns indicate that a nonlinear model may achieve better performance.

Recommended next step:

* Train and evaluate a Random Forest Regressor.
* Compare MAE, RMSE, and R² against the linear baseline.
* Determine whether nonlinear feature interactions improve prediction accuracy.

### Generated Reports

* `reports/metrics_linear.json`
* `reports/linear_diagnostics.md`

### Generated Figures

* `reports/figures/residuals_linear.png`
* `reports/figures/residuals_vs_humidity.png`



## Reproducibility

### Saved Artifacts

```text
models/
├── random_forest.joblib
├── minmax_scaler_train.joblib
└── feature_cols.json
```

### Environment

* Python 3.12
* pandas
* numpy
* scikit-learn
* joblib
* matplotlib

### Run Inference

```bash
python src/predict.py
```

### Example Output

```text
Predicted Yield: 17.89 kg
```

### Reproducibility Notes

* Random Forest model is stored in `models/random_forest.joblib`.
* Feature scaling uses the saved `MinMaxScaler`.
* Model inference does not require retraining.
* Feature order is preserved through `feature_cols.json`.
* All model artifacts are loaded from the `models/` directory.


## Run Locally

1. Activate virtual environment

```bash
venv\Scripts\activate

## Live Demo

https://mushroom-yield-forecast-apjbqtsnkmpwcwqbbbzmcz.streamlit.app

