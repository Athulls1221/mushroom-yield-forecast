\# Cleaning Log



Dataset: Polyhouse Sensor Data



\## Missing Values

Generated using:



df.isna().sum()



\## Validation Rules



Temperature:

10°C - 35°C



Humidity:

50% - 100%



CO2:

400 ppm - 2000 ppm



\## Missing Data Handling



Forward-fill sensor columns up to 2 consecutive records.



\## Target Variable



Removed rows where yield\_kg is missing.



\## Duplicate Handling



Removed duplicate timestamps using keep="last".



\## Output



Saved cleaned dataset:



data/interim/02\_cleaned.parquet

