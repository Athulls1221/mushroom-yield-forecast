import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# Load test data
X_test = np.load("data/proccessed/X_test.npy")
y_test = np.load("data/proccessed/y_test.npy")

# Load trained model
model = joblib.load("models/linear_regression.joblib")

# Predict
pred_test = model.predict(X_test)

# Residuals
residuals = y_test - pred_test

# Create folder
os.makedirs("reports/figures", exist_ok=True)

# ----------------------------------
# Plot 1: Residuals vs Predicted
# ----------------------------------

plt.figure(figsize=(6,4))
plt.scatter(pred_test, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted Yield (kg)")
plt.ylabel("Residual (kg)")
plt.title("Residuals vs Predicted Yield")

plt.tight_layout()
plt.savefig(
    "reports/figures/residuals_linear.png",
    dpi=150
)

plt.close()

# ----------------------------------
# Plot 2: Residuals vs Humidity
# ----------------------------------

plt.figure(figsize=(6,4))
plt.scatter(X_test[:,1], residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Scaled Humidity")
plt.ylabel("Residual (kg)")
plt.title("Residuals vs Humidity")

plt.tight_layout()
plt.savefig(
    "reports/figures/residuals_vs_humidity.png",
    dpi=150
)

plt.close()

# ----------------------------------
# Diagnostics Report
# ----------------------------------

with open(
    "reports/linear_diagnostics.md",
    "w"
) as f:

    f.write("# Linear Regression Diagnostics\n\n")

    f.write("## Findings\n\n")

    f.write(
        "- Residuals were computed as actual minus predicted values.\n"
    )

    f.write(
        "- Residual plots were inspected for systematic patterns.\n"
    )

    f.write(
        "- Most residuals are centered around zero, indicating the model captures part of the signal.\n"
    )

    f.write(
        "- Some spread remains unexplained, suggesting additional features or nonlinear relationships may exist.\n"
    )

    f.write(
        "- R² from the baseline model was approximately 0.427.\n"
    )

    f.write(
        "- Linear regression provides a useful baseline but does not explain all yield variation.\n\n"
    )

    f.write("## Recommendation\n\n")

    f.write(
        "Proceed to a nonlinear model such as Random Forest to determine whether predictive performance improves over the linear baseline."
    )

print("Diagnostics completed.")