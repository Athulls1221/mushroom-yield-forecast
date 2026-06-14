\# Cross Validation Results



\## Methodology



TimeSeriesSplit with 5 splits was used to evaluate model stability while preserving temporal order. Only training data was used during cross-validation.



\## Results



| Model             | Mean CV MAE | CV Std |

| ----------------- | ----------- | ------ |

| Linear Regression | 0.4406      | 0.0338 |

| Random Forest     | 0.4748      | 0.0575 |



\## Interpretation



Linear Regression achieved a lower cross-validated MAE than Random Forest, indicating better predictive performance on unseen data.



Linear Regression also showed lower variance across folds, suggesting more stable and consistent behavior.



\## Overfitting Analysis



Linear Regression:



\* Test MAE = 0.4194

\* CV MAE = 0.4406



Random Forest:



\* Test MAE = 0.4503

\* CV MAE = 0.4748



The differences between cross-validation and test errors are small for both models. Therefore, there is no significant evidence of overfitting.



\## Conclusion



Linear Regression remains the preferred model because it achieves lower error, better stability, and greater interpretability than Random Forest on this dataset.



