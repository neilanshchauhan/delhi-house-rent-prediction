"""
Feature Engineering Pipeline for Delhi House Rent Prediction
=============================================================
This script performs feature engineering on cleaned housing data:
- Filters realistic bedroom counts
- Handles rare localities
- Creates derived features (price_per_sqft, location_avg_price)
- Encodes and scales selected features (others passed through)
- Saves preprocessor for inference

Usage:
    python feature_engineering.py \
        --input data/processed/cleaned_data_v1.csv \
        --output data/processed/featured_data.csv \
        --preprocessor models/preprocessor.pkl
"""
import os
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')


def preprocess_features(df):
    """
    Apply all feature preprocessing steps in one function.

    Steps:
    1. Filter bedroom outliers (1 < bedrooms < 9)
    2. Clean locality names (strip whitespace)
    3. Consolidate rare localities (<=10 occurrences → 'other')
    4. Create price_per_sqft feature
    5. Create location_avg_price feature

    Args:
        df: Input DataFrame with cleaned data

    Returns:
        DataFrame with engineered features
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input shape: {df.shape}")

    df_processed = df.copy()

    # 1. Filter bedroom outliers
    initial_rows = df_processed.shape[0]
    df_processed = df_processed[(df_processed['bedrooms'] > 1) & (df_processed['bedrooms'] < 9)]
    removed = initial_rows - df_processed.shape[0]
    logger.info(f"✅ Filtered {removed} rows with unrealistic bedroom counts")

    # 2. Clean locality names
    df_processed['localityName'] = df_processed['localityName'].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    logger.info(f"✅ Cleaned locality names: {df_processed['localityName'].nunique()} unique localities")

    # 3. Consolidate rare localities
    location_count = df_processed['localityName'].value_counts()
    rare_index = location_count[location_count <= 10].index
    df_processed['localityName'] = df_processed['localityName'].apply(
        lambda x: 'other' if x in rare_index else x
    )
    logger.info(f"✅ Consolidated {len(rare_index)} rare localities into 'other'")
    logger.info(f"   Final unique localities: {df_processed['localityName'].nunique()}")

    # 4. Create price_per_sqft
    df_processed['price_per_sqft'] = np.where(
        df_processed['size_sq_ft'] > 0,
        df_processed['price'] / df_processed['size_sq_ft'],
        np.nan
    )
    null_count = df_processed['price_per_sqft'].isnull().sum()
    if null_count > 0:
        logger.warning(f"⚠️  Created {null_count} NaN values in price_per_sqft")
    # guard min/max if all NaN
    if df_processed['price_per_sqft'].notnull().any():
        logger.info(f"✅ Created price_per_sqft feature (range: {df_processed['price_per_sqft'].min():.2f} - {df_processed['price_per_sqft'].max():.2f})")
    else:
        logger.info("✅ Created price_per_sqft feature (all NaN)")

    # 5. Create location_avg_price (recompute after consolidation)
    location_price = df_processed.groupby('localityName')['price'].mean()
    df_processed['location_avg_price'] = df_processed['localityName'].map(location_price)
    if df_processed['location_avg_price'].notnull().any():
        logger.info(f"✅ Created location_avg_price feature (range: {df_processed['location_avg_price'].min():.2f} - {df_processed['location_avg_price'].max():.2f})")
    else:
        logger.info("✅ Created location_avg_price feature (all NaN)")

    logger.info(f"Output shape: {df_processed.shape}")
    logger.info(f"Features: {list(df_processed.columns)}")
    logger.info("=" * 60)

    return df_processed


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """Custom Label Encoder that handles unknown categories during transform."""

    def __init__(self):
        self.encoders = {}
        self.default_class_for_col = {}

    def fit(self, X, y=None):
        """Fit label encoders for each column."""
        # Accept numpy array or DataFrame
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        for col in X_df.columns:
            ser = X_df[col].astype(str).fillna('nan')
            le = LabelEncoder()
            le.fit(ser)
            self.encoders[col] = le
            # Choose most frequent class as fallback
            most_freq = ser.mode().iloc[0] if not ser.mode().empty else le.classes_[0]
            self.default_class_for_col[col] = most_freq

        return self

    def transform(self, X):
        """Transform using fitted encoders, handling unknown values."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        out_cols = []
        for col in X_df.columns:
            if col not in self.encoders:
                raise ValueError(f"Column '{col}' was not seen in fit for CustomLabelEncoder")
            le = self.encoders[col]
            default = self.default_class_for_col[col]
            # convert to str and replace unknowns with default
            ser = X_df[col].astype(str).fillna('nan').apply(lambda x: x if x in le.classes_ else default)
            transformed = le.transform(ser)
            out_cols.append(transformed.reshape(-1, 1))

        # concatenate columns side-by-side
        result = np.hstack(out_cols)
        return result

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        return self.fit(X, y).transform(X)


def create_preprocessor():
    """
    Create a preprocessing pipeline for encoding and scaling.

    Handles:
    - Numerical features: Imputation + StandardScaler
    - Categorical features: Label encoding with unknown handling

    Returns:
        ColumnTransformer pipeline
    """
    logger.info("Creating preprocessor pipeline...")

    # Define feature groups
    # Only transform/select features that need it. 'bedrooms' will be passed through unchanged.
    categorical_features = ['localityName', 'propertyType']
    numerical_features = ['size_sq_ft', 'location_avg_price', 'price_per_sqft']

    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('label', CustomLabelEncoder())
    ])

    # Combine preprocessors in a column transformer
    # Use remainder='passthrough' so columns like 'bedrooms' remain unchanged and are included in the transformed output
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    logger.info("✅ Preprocessor pipeline created successfully")

    # attach feature lists for downstream naming convenience
    preprocessor.numerical_features = numerical_features
    preprocessor.categorical_features = categorical_features

    return preprocessor


def validate_data(df):
    """Validate input data before processing."""
    logger.info("Validating input data...")

    required_columns = ['size_sq_ft', 'propertyType', 'bedrooms', 'localityName', 'price']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.shape[0] == 0:
        raise ValueError("Input dataframe is empty")

    logger.info("✅ Data validation passed")


def _safe_makedirs(path):
    """Make directory if path not empty."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def run_feature_engineering(input_file, output_file, preprocessor_file):
    """
    Full feature engineering pipeline.

    Steps:
    1. Load cleaned data
    2. Validate data
    3. Preprocess features (filter, clean, create features)
    4. Create and fit preprocessor (encode + scale)
    5. Transform features
    6. Save preprocessor and complete dataset (X_transformed + y)

    Args:
        input_file: Path to cleaned CSV file
        output_file: Path for output CSV file (transformed X + y)
        preprocessor_file: Path for saving the preprocessor (.pkl)
    """
    try:
        # Create output directories (if provided)
        _safe_makedirs(os.path.dirname(output_file))
        _safe_makedirs(os.path.dirname(preprocessor_file))

        # Load cleaned data
        logger.info("=" * 60)
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")

        # Validate data
        validate_data(df)

        # Preprocess features (all feature engineering steps)
        df_processed = preprocess_features(df)
        logger.info(f"✅ Feature preprocessing complete: {df_processed.shape}")

        # Separate features and target
        X = df_processed.drop(columns=['price'])
        y = df_processed['price']

        logger.info(f"Features (X) shape: {X.shape}")
        logger.info(f"Target (y) shape: {y.shape}")

        # Create and fit the preprocessor
        preprocessor = create_preprocessor()

        logger.info("Fitting preprocessor and transforming features...")
        X_transformed = preprocessor.fit_transform(X)
        logger.info(f"✅ Transformed features shape: {X_transformed.shape}")

        # Save the preprocessor
        joblib.dump(preprocessor, preprocessor_file)
        logger.info(f"✅ Saved preprocessor to {preprocessor_file}")

        # Create output dataframe with transformed features + target
        # Use feature lists from preprocessor to avoid mismatch
        numerical_feats = getattr(preprocessor, 'numerical_features', [])
        categorical_feats = getattr(preprocessor, 'categorical_features', [])
        # Passthrough features are those in X.columns not transformed explicitly
        passthrough_feats = [c for c in X.columns if c not in (numerical_feats + categorical_feats)]

        # Construct feature names respecting ColumnTransformer output order: transformed groups first, then passthrough columns
        # IMPORTANT: preserve original feature names (no prefixes)
        feature_names = numerical_feats + categorical_feats + passthrough_feats

        if X_transformed.shape[1] != len(feature_names):
            raise ValueError(f"Transformed feature count ({X_transformed.shape[1]}) does not match feature_names length ({len(feature_names)}).")

        df_final = pd.DataFrame(
            X_transformed,
            columns=feature_names,
            index=y.index
        )

        # Add target variable (NOT transformed)
        df_final['price'] = y.values

        # Save complete dataset
        df_final.to_csv(output_file, index=False)
        logger.info(f"✅ Saved complete dataset to {output_file}")
        logger.info(f"   Columns: {list(df_final.columns)}")

        # Summary statistics
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Preprocessor file: {preprocessor_file}")
        logger.info(f"Original shape: {df.shape}")
        logger.info(f"Final shape: {df_final.shape}")
        logger.info(f"Features (transformed): {X_transformed.shape[1]}")
        logger.info(f"Target variable: price (NOT transformed)")
        logger.info("=" * 60)

        return df_final

    except Exception as e:
        logger.error(f"❌ Error in feature engineering pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Feature engineering for Delhi housing rent prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python feature_engineering.py \\
        --input data/processed/cleaned_data_v1.csv \\
        --output data/processed/featured_data.csv \\
        --preprocessor models/preprocessor.pkl
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to cleaned CSV file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path for output CSV file (transformed X + y)'
    )
    parser.add_argument(
        '--preprocessor',
        required=True,
        help='Path for saving the preprocessor (.pkl file)'
    )

    args = parser.parse_args()

    # Run the pipeline
    run_feature_engineering(args.input, args.output, args.preprocessor)
