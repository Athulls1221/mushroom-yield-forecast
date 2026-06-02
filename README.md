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

