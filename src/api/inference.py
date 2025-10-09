# src/api/inference.py

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# These will be created in the next steps, names are updated for our project
from .schemas import RentPredictionRequest, PredictionResponse

# --- 1. Load All Necessary Artifacts ---

# This script will now load the entire cleaned dataset to perform calculations,
# as requested. This is memory-intensive and not a standard production practice,
# but it fulfills the project's current goal.

# Determine the base directory of the current file to build absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the model, preprocessor, and the data file for on-the-fly calculations
MODEL_PATH = os.path.join(BASE_DIR, "../../models/final/delhi_house_rent_predictor.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "../../models/preprocessor.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../../data/interim/cleaned_house_data_v1.csv") # Path to the data for calculations

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    # Load the full dataset for feature calculation
    df_train = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model, preprocessor, or data file: {str(e)}")

# --- 2. Define Prediction Functions ---

def predict_rent(request: RentPredictionRequest) -> PredictionResponse:
    """
    Predicts a single house rent by calculating features on the fly.
    """
    # Create a DataFrame from the single request
    input_data = pd.DataFrame([request.model_dump()])

    # --- On-the-fly Feature Engineering (The "Hack") ---
    location = request.localityName
    
    # Filter the training dataframe for the requested location
    df_loc = df_train[df_train['localityName'] == location]
    
    if not df_loc.empty:
        # Calculate features based on the location's historical data
        location_avg_price = df_loc['price'].mean()
        # Using the exact logic from your Flask app
        price_per_sqft = df_loc['price'].mean() / df_loc['size_sq_ft'].mean()
    else:
        # Fallback for new locations: use global average
        location_avg_price = df_train['price'].mean()
        price_per_sqft = df_train['price'].mean() / df_train['size_sq_ft'].mean()

    input_data['location_avg_price'] = location_avg_price
    input_data['price_per_sqft'] = price_per_sqft
    
    # Ensure the column order matches what the preprocessor was trained on
    training_columns = preprocessor.feature_names_in_
    input_data = input_data[training_columns]

    # Preprocess the input data
    processed_features = preprocessor.transform(input_data)

    # Make the prediction
    predicted_rent = model.predict(processed_features)[0]
    
    # Format the response
    predicted_rent = round(float(predicted_rent), 2)
    lower_bound = round(predicted_rent * 0.9, 2)
    upper_bound = round(predicted_rent * 1.1, 2)

    return PredictionResponse(
        predicted_rent=predicted_rent,
        confidence_interval=[lower_bound, upper_bound],
        prediction_time=datetime.now().isoformat()
    )


def batch_predict_rent(requests: list[RentPredictionRequest]) -> list[PredictionResponse]:
    """
    Performs batch predictions. This is an optimized version for multiple requests.
    """
    input_df = pd.DataFrame([req.model_dump() for req in requests])
    
    # --- Optimized On-the-fly Feature Engineering for Batches ---
    # Pre-calculate maps from the training data once
    price_map = df_train.groupby('localityName')['price'].mean()
    pps_map = (df_train.groupby('localityName')['price'].mean() / 
               df_train.groupby('localityName')['size_sq_ft'].mean())

    # Use maps to create features for the entire batch
    input_df['location_avg_price'] = input_df['localityName'].map(price_map).fillna(df_train['price'].mean())
    input_df['price_per_sqft'] = input_df['localityName'].map(pps_map).fillna(df_train['price'].mean() / df_train['size_sq_ft'].mean())
    
    # Ensure column order
    training_columns = preprocessor.feature_names_in_
    input_df = input_df[training_columns]

    # Preprocess and predict in a batch
    processed_features = preprocessor.transform(input_df)
    predicted_rents = model.predict(processed_features)

    # Format each prediction into a PredictionResponse object
    response_list = []
    for rent in predicted_rents:
        rent = round(float(rent), 2)
        lower_bound = round(rent * 0.9, 2)
        upper_bound = round(rent * 1.1, 2)
        response_list.append(PredictionResponse(
            predicted_rent=rent,
            confidence_interval=[lower_bound, upper_bound],
            prediction_time=datetime.now().isoformat()
        ))
        
    return response_list