# Linear Regression Diagnostics

## Findings

- Residuals were computed as actual minus predicted values.
- Residual plots were inspected for systematic patterns.
- Most residuals are centered around zero, indicating the model captures part of the signal.
- Some spread remains unexplained, suggesting additional features or nonlinear relationships may exist.
- R² from the baseline model was approximately 0.427.
- Linear regression provides a useful baseline but does not explain all yield variation.

## Recommendation

Proceed to a nonlinear model such as Random Forest to determine whether predictive performance improves over the linear baseline.