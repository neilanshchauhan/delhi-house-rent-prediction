# src/api/inference.py

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .schemas import RentPredictionRequest, PredictionResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load all artifacts with proper error handling
MODEL_PATH = "models/trained/delhi_house_rent_predictor.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"
# Use the final_data.csv which should have the base features
DATA_PATH = "data/processed/final_data.csv"

try:
    # Check if files exist
    for path, name in [(MODEL_PATH, "Model"), (PREPROCESSOR_PATH, "Preprocessor"), (DATA_PATH, "Data")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
    
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    df_train = pd.read_csv(DATA_PATH)
    
    logger.info("All artifacts loaded successfully")
    logger.info(f"Training data shape: {df_train.shape}")
    logger.info(f"Training data columns: {df_train.columns.tolist()}")
    
except Exception as e:
    logger.error(f"Error loading artifacts: {str(e)}")
    raise RuntimeError(f"Error loading artifacts: {str(e)}")


def predict_rent(request: RentPredictionRequest) -> PredictionResponse:
    """Single prediction endpoint"""
    try:
        # Create input dataframe with the exact features expected
        input_data = pd.DataFrame([{
            'size_sq_ft': request.size_sq_ft,
            'bedrooms': request.bedrooms,
            'localityName': request.localityName,
            'propertyType': request.propertyType
        }])
        
        logger.info(f"Input data: {input_data.to_dict()}")
        
        # --- Create leaky features (matching engineer.py logic) ---
        location = request.localityName
        
        # Check if location exists in training data
        if location in df_train['localityName'].values:
            df_loc = df_train[df_train['localityName'] == location]
            location_avg_price = df_loc['price'].mean()
            # Calculate average price per sqft for this location
            avg_price_per_sqft = (df_loc['price'] / df_loc['size_sq_ft']).mean()
        else:
            # Fallback for unknown locations - use overall averages
            logger.warning(f"Unknown location: {location}. Using overall averages.")
            location_avg_price = df_train['price'].mean()
            avg_price_per_sqft = (df_train['price'] / df_train['size_sq_ft']).mean()
        
        # Add the engineered features
        input_data['location_avg_price'] = location_avg_price
        input_data['price_per_sqft'] = request.size_sq_ft * avg_price_per_sqft / request.size_sq_ft  # This should be current property's estimated price/sqft
        
        # Actually, for inference, we should use the request's size to calculate price_per_sqft
        # But since we don't have the price yet (that's what we're predicting), 
        # we use the location's average price_per_sqft
        input_data['price_per_sqft'] = avg_price_per_sqft
        
        logger.info(f"Features after engineering: {input_data.columns.tolist()}")
        logger.info(f"Feature values: {input_data.iloc[0].to_dict()}")
        
        # Ensure columns are in the right order for the preprocessor
        expected_features = ['size_sq_ft', 'location_avg_price', 'price_per_sqft', 
                           'localityName', 'propertyType', 'bedrooms']
        
        # Reorder columns to match training
        input_data = input_data[expected_features]
        
        # Preprocess and predict
        processed_features = preprocessor.transform(input_data)
        predicted_rent = model.predict(processed_features)[0]
        
        # Format response
        predicted_rent = round(float(predicted_rent), 2)
        lower_bound = round(predicted_rent * 0.9, 2)
        upper_bound = round(predicted_rent * 1.1, 2)

        return PredictionResponse(
            predicted_rent=predicted_rent,
            confidence_interval=[lower_bound, upper_bound],
            prediction_time=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in predict_rent: {str(e)}")
        raise


def batch_predict_rent(requests: list[RentPredictionRequest]) -> list[PredictionResponse]:
    """Batch prediction endpoint"""
    try:
        # Create input dataframe
        input_df = pd.DataFrame([{
            'size_sq_ft': req.size_sq_ft,
            'bedrooms': req.bedrooms,
            'localityName': req.localityName,
            'propertyType': req.propertyType
        } for req in requests])
        
        # Pre-compute location statistics
        price_map = df_train.groupby('localityName')['price'].mean().to_dict()
        pps_map = df_train.groupby('localityName').apply(
            lambda x: (x['price'] / x['size_sq_ft']).mean()
        ).to_dict()
        
        # Default values for unknown locations
        default_price = df_train['price'].mean()
        default_pps = (df_train['price'] / df_train['size_sq_ft']).mean()
        
        # Apply engineered features
        input_df['location_avg_price'] = input_df['localityName'].map(price_map).fillna(default_price)
        input_df['price_per_sqft'] = input_df['localityName'].map(pps_map).fillna(default_pps)
        
        # Ensure columns are in the right order
        expected_features = ['size_sq_ft', 'location_avg_price', 'price_per_sqft', 
                           'localityName', 'propertyType', 'bedrooms']
        input_df = input_df[expected_features]
        
        # Preprocess and predict
        processed_features = preprocessor.transform(input_df)
        predicted_rents = model.predict(processed_features)

        # Create responses
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
    
    except Exception as e:
        logger.error(f"Error in batch_predict_rent: {str(e)}")
        raise