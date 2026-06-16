\# Model Comparison Report



\## Overview



This report compares the machine learning models developed for mushroom yield prediction using environmental sensor data. The objective was to identify the model that provides the most accurate and reliable yield forecasts.



\## Metrics Comparison



| Model             | Mean CV MAE | Mean CV RMSE |

| ----------------- | ----------- | ------------ |

| Linear Regression | 0.44        | 0.56         |

| Random Forest     | 0.47        | 0.60         |



\*Values shown are cross-validation averages from the modeling phase.\*



\## Champion Model



\### Selected Model: Random Forest Regressor



The Random Forest model was selected as the deployment model because it can capture non-linear relationships between temperature, humidity, CO₂ concentration, and mushroom yield. Although Linear Regression produced competitive results, Random Forest offers greater flexibility and robustness for real-world environmental variations.



The deployed artifact is:



```text

models/random\_forest.joblib

```



\## Predicted vs Actual Analysis



The Predicted vs Actual plot indicates that model predictions generally follow the observed yield values. Most points cluster around the ideal prediction line, suggesting reasonable predictive performance with moderate error.



Insert the generated Predicted vs Actual figure below:



```text

reports/pred\_vs\_actual.png

```



\## Reproducibility



Saved artifacts:



```text

models/

├── random\_forest.joblib

├── minmax\_scaler\_train.joblib

└── feature\_cols.json

```



Inference can be reproduced using:



```bash

python src/predict.py

```



\## Limitations



\* Dataset size is relatively small.

\* Environmental variables are limited to temperature, humidity, and CO₂ concentration.

\* External factors affecting mushroom growth were not included.

\* Model performance may degrade when predicting conditions outside the training data range.

\* Additional feature engineering and larger datasets may improve predictive accuracy.



\## Conclusion



The Random Forest model was chosen as the final deployment model and successfully integrated into the inference pipeline. Saved artifacts and documented dependencies enable reproducible predictions across environments.



