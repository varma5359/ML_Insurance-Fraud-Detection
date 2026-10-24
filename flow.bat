
@echo off

mkdir data
mkdir src
mkdir src\data_processing
mkdir src\modeling

type nul > src\data_processing\__init__.py
type nul > src\data_processing\cleaning.py
type nul > src\data_processing\preprocessing.py

type nul > src\modeling\__init__.py
type nul > src\modeling\training.py
type nul > src\modeling\evaluation.py
type nul > src\modeling\prediction.py

type nul > Insurance_Fraud_Detection.ipynb
type nul > app.py
type nul > requirements.txt
type nul > README.md

echo.
echo Project structure created successfully.
echo No machine learning code has been added.
pause
