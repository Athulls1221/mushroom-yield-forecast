# Random Forest Conclusion

A Random Forest Regressor was trained and evaluated on the mushroom yield dataset.

The model achieved:

* MAE = 0.4503
* RMSE = 0.5803
* R² = 0.3265

When compared with the Linear Regression baseline:

* Linear Regression MAE = 0.4194
* Linear Regression RMSE = 0.5352
* Linear Regression R² = 0.4272

The Random Forest model performed worse on all evaluation metrics. This suggests that the relationship between the available sensor variables and mushroom yield is relatively simple and can be modeled effectively using Linear Regression.

Therefore, the additional complexity of Random Forest is not justified for the current dataset, and Linear Regression remains the preferred model.
