# Devixo-Solutions-Task-04
House Price Prediction: end-to-end ML application with Streamlit.
# House Price Prediction – Devixo Solutions Task 04

An end-to-end machine learning application that predicts house prices from
property details, built with Scikit-learn and deployed with Streamlit.

http://localhost:8501/

## Dataset
House price dataset with 2,000 records and 10 columns
(Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, Price).
Target variable: Price.

## Workflow
1. Data loading and exploration
2. Missing value and duplicate checks, outlier detection (IQR)
3. Feature engineering (HouseAge, TotalRooms, AreaPerBedroom, BathBedRatio, AreaPerFloor, IsNew, AgeGroup)
4. One-hot encoding and StandardScaler
5. Training and comparing 5 regression models
6. Hyperparameter tuning (RandomizedSearchCV and GridSearchCV)
7. Saving the final model and building the Streamlit app

## Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| (fill from your results table) | | | |

Final model: **MODEL_NAME** (R² = X.XX on the test set)

## Run locally
    pip install -r requirements.txt
    python -m streamlit run app.py

## Files
- `app.py`: Streamlit application
- `*.pkl`: saved model, scaler, feature columns and metadata
- `House_Price_Prediction.ipynb`: full notebook
- `data/`: dataset
- `screenshots/`: app and notebook screenshots

## Screenshots


## Author
Wajiha Waqar – BS Robotics and Intelligent Systems, NUML Islamabad
