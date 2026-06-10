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
