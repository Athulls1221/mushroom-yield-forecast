\# EDA Findings



\## Correlation Analysis



A correlation heatmap was generated for temperature, humidity, CO₂ concentration, and mushroom yield.



The heatmap was used to identify relationships between environmental conditions and yield.



\## Scatter Plot Analysis



Three scatter plots were created:



\* Humidity (%) vs Yield (kg)

\* Temperature (°C) vs Yield (kg)

\* CO₂ (ppm) vs Yield (kg)



These visualizations help identify trends, clusters, and potential nonlinear relationships.



\## Key Takeaways



\* Humidity appears to have a noticeable relationship with mushroom yield.

\* Temperature may influence yield, but the relationship should be validated using modeling.

\* CO₂ concentration may also affect yield and should be considered as a predictive feature.

\* Scatter plots provide a better understanding of relationships than correlation values alone.

\* Correlation does not imply causation; observed relationships may be influenced by other environmental factors.



\## Conclusion



The EDA phase provided insight into the relationships between sensor measurements and mushroom yield. These findings will guide feature selection and model development in the next phase of the project.

# EDA Findings

## Scatter Plots Created

The following scatter plots were generated and saved as PNG files:

* humidity_vs_yield.png
* temperature_vs_yield.png
* co2_vs_yield.png

A correlation heatmap was also generated:

* corr_heatmap.png

## Nonlinear Patterns and Clusters

* The scatter plots were inspected for trends, clusters, and unusual observations.
* No obvious extreme outliers were observed in the dataset.
* Some relationships may not be perfectly linear, indicating that environmental variables can affect mushroom growth in complex ways.
* Scatter plots provide additional insight beyond simple correlation coefficients.

## Mushroom Biology Takeaways

* Humidity is an important environmental factor because mushrooms require high moisture levels for healthy growth and fruiting.
* Temperature influences metabolic activity and can affect mushroom yield when conditions move away from the optimal growing range.
* CO₂ concentration impacts gas exchange within the growing environment and may influence development and harvest weight.
* Yield appears to be related to multiple environmental factors rather than a single sensor measurement.
* Environmental variables should be retained as candidate features for predictive modeling in later project phases.

## Caveat

Correlation does not imply causation. Relationships observed during EDA suggest possible associations, but additional modeling and experimentation are required to confirm causal effects.


