# src/features/engineer.py

import os
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys

# Add project root to path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom class from its central location
from src.api.utils import CustomLabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('feature-engineering')

def create_preprocessor():
    """Creates the preprocessing pipeline."""
    categorical_features = ['localityName', 'propertyType']
    numerical_features = ['size_sq_ft', 'location_avg_price', 'price_per_sqft']
    
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('label', CustomLabelEncoder())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # 'bedrooms' will be passed through
    )
    # Attach feature lists for downstream naming convenience
    preprocessor.numerical_features = numerical_features
    preprocessor.categorical_features = categorical_features
    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """
    Loads data, creates leaky features, fits and saves the preprocessor,
    and saves the final processed dataset.
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # --- Leaky Feature Engineering ---
    # This logic creates features derived from the target variable.
    df['price_per_sqft'] = df['price'] / df['size_sq_ft']
    location_price_map = df.groupby('localityName')['price'].mean()
    df['location_avg_price'] = df['localityName'].map(location_price_map)
    df.dropna(inplace=True)
    logger.info("✅ Leaky features created for training.")
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    preprocessor = create_preprocessor()
    
    logger.info("Fitting preprocessor")
    preprocessor.fit(X)
    
    os.makedirs(os.path.dirname(preprocessor_file), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"✅ Preprocessor (trained on leaky data) saved to {preprocessor_file}")

    # --- Transform the data and save it ---
    logger.info("Transforming data and saving the final featured dataset...")
    X_transformed = preprocessor.transform(X)

    # Reconstruct the DataFrame with correct column names
    num_feats = preprocessor.numerical_features
    cat_feats = preprocessor.categorical_features
    passthrough_feats = [col for col in X.columns if col not in num_feats and col not in cat_feats]
    
    # The order must match the ColumnTransformer's output
    final_columns = num_feats + cat_feats + passthrough_feats

    df_final = pd.DataFrame(X_transformed, columns=final_columns, index=y.index)
    df_final['price'] = y.values

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    logger.info(f"✅ Final featured dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file (e.g., final_data.csv)')
    parser.add_argument('--output', required=True, help='Path to save the final processed CSV file')
    parser.add_argument('--preprocessor', required=True, help='Path for saving the preprocessor .pkl file')
    args = parser.parse_args()
    run_feature_engineering(args.input, args.output, args.preprocessor)