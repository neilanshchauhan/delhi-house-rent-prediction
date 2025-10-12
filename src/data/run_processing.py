"""
Data Preprocessing Pipeline for Delhi House Rent Prediction
==========================================================
This script performs complete data preprocessing in 2 stages:

Stage 1 (Basic): Column removal, missing value checks
Stage 2 (Advanced): Outlier filtering, feature engineering

The script outputs a single final_data.csv file with all transformations applied.

Usage:
    python src/data/preprocessing.py \
        --input data/raw/data.csv \
        --output data/processed/final_data.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-preprocessing')


def load_data(file_path):
    """
    Load raw data from a CSV file.
    
    Args:
        file_path: Path to the raw CSV file
        
    Returns:
        DataFrame with raw data
    """
    logger.info(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def drop_unnecessary_columns(df):
    """
    Stage 1: Remove columns that are not needed for modeling.
    
    These columns include:
    - Index columns (Unnamed: 0)
    - Geographic coordinates (latitude, longitude)
    - General metadata (cityName, companyName)
    - Distance features (closest_metro_station_km, etc.)
    - Suburb information (suburbName)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unnecessary columns removed
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: BASIC PREPROCESSING")
    logger.info("=" * 60)
    logger.info("Removing unnecessary columns...")
    
    # Define columns to drop
    columns_to_drop = [
        'Unnamed: 0',
        'latitude',
        'longitude',
        'cityName',
        'companyName',
        'closest_mtero_station_km',
        'AP_dist_km',
        'Aiims_dist_km',
        'NDRLW_dist_km',
        'suburbName'
    ]
    
    # Only drop columns that exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        df_cleaned = df.drop(columns=existing_columns_to_drop)
        logger.info(f"✅ Dropped {len(existing_columns_to_drop)} columns:")
        for col in existing_columns_to_drop:
            logger.info(f"   - {col}")
    else:
        df_cleaned = df.copy()
        logger.warning("No columns to drop found in dataframe")
    
    logger.info(f"Remaining columns: {list(df_cleaned.columns)}")
    logger.info(f"Shape after column removal: {df_cleaned.shape}")
    
    return df_cleaned


def check_missing_values(df):
    """
    Check and log missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame (unchanged, for informational purposes)
    """
    logger.info("Checking for missing values...")
    
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        logger.warning(f"Found {total_missing} missing values in dataset:")
        for column, count in missing_values[missing_values > 0].items():
            percentage = (count / len(df)) * 100
            logger.warning(f"   {column}: {count} ({percentage:.2f}%)")
    else:
        logger.info("✅ No missing values found in dataset")
    
    return df


def validate_data(df):
    """
    Validate that the dataset meets basic requirements.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame (unchanged)
    """
    logger.info("Validating dataset...")
    
    # Check if dataframe is empty
    if df.shape[0] == 0:
        raise ValueError("Dataset is empty!")
    
    # Check for required columns
    required_columns = ['size_sq_ft', 'propertyType', 'bedrooms', 'localityName', 'price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"✅ Dataset validation passed")
    logger.info(f"   Rows: {df.shape[0]}")
    logger.info(f"   Columns: {df.shape[1]}")
    
    return df


def filter_bedroom_outliers(df):
    """
    Stage 2: Filter properties with realistic bedroom counts (1 < bedrooms < 9).
    
    Removes properties with 1 or fewer bedrooms and 9 or more bedrooms.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: ADVANCED PREPROCESSING")
    logger.info("=" * 60)
    logger.info("Filtering bedroom outliers (1 < bedrooms < 9)...")
    
    initial_rows = df.shape[0]
    df_filtered = df[(df['bedrooms'] > 1) & (df['bedrooms'] < 9)].copy()
    removed = initial_rows - df_filtered.shape[0]
    
    logger.info(f"✅ Filtered {removed} rows with unrealistic bedroom counts")
    logger.info(f"   Remaining rows: {df_filtered.shape[0]}")
    
    return df_filtered


def clean_locality_names(df):
    """
    Clean locality names by removing whitespace.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned locality names
    """
    logger.info("Cleaning locality names (removing whitespace)...")
    
    df_cleaned = df.copy()
    df_cleaned['localityName'] = df_cleaned['localityName'].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    
    logger.info(f"✅ Cleaned locality names: {df_cleaned['localityName'].nunique()} unique localities")
    
    return df_cleaned


def consolidate_rare_localities(df, threshold=10):
    """
    Consolidate localities with rare occurrences (<=threshold) into 'other'.
    
    Reduces cardinality of locality feature by grouping infrequent categories.
    
    Args:
        df: Input DataFrame
        threshold: Maximum occurrence count to be considered rare (default: 10)
        
    Returns:
        DataFrame with consolidated localities
    """
    logger.info(f"Consolidating rare localities (occurrences <= {threshold})...")
    
    df_consolidated = df.copy()
    location_count = df_consolidated['localityName'].value_counts()
    initial_unique = len(location_count)
    
    rare_localities = location_count[location_count <= threshold].index
    df_consolidated['localityName'] = df_consolidated['localityName'].apply(
        lambda x: 'other' if x in rare_localities else x
    )
    
    final_unique = df_consolidated['localityName'].nunique()
    
    logger.info(f"✅ Consolidated {len(rare_localities)} rare localities into 'other'")
    logger.info(f"   Unique localities reduced from {initial_unique} to {final_unique}")
    
    return df_consolidated


def create_price_per_sqft(df):
    """
    Create price_per_sqft feature (handling division by zero).
    
    Calculates price per square foot for each property.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with price_per_sqft column added
    """
    logger.info("Creating price_per_sqft feature...")
    
    df_featured = df.copy()
    df_featured['price_per_sqft'] = np.where(
        df_featured['size_sq_ft'] > 0,
        df_featured['price'] / df_featured['size_sq_ft'],
        np.nan
    )
    
    null_count = df_featured['price_per_sqft'].isnull().sum()
    if null_count > 0:
        logger.warning(f"⚠️  Created {null_count} NaN values in price_per_sqft (zero size_sq_ft)")
    
    if df_featured['price_per_sqft'].notnull().any():
        min_pps = df_featured['price_per_sqft'].min()
        max_pps = df_featured['price_per_sqft'].max()
        logger.info(f"✅ Created price_per_sqft feature")
        logger.info(f"   Range: {min_pps:.2f} - {max_pps:.2f}")
    
    return df_featured


def create_location_avg_price(df):
    """
    Create location_avg_price feature.
    
    Calculates average price for each locality (used as proxy for location value).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with location_avg_price column added
    """
    logger.info("Creating location_avg_price feature...")
    
    df_featured = df.copy()
    location_price = df_featured.groupby('localityName')['price'].mean()
    df_featured['location_avg_price'] = df_featured['localityName'].map(location_price)
    
    if df_featured['location_avg_price'].notnull().any():
        min_lap = df_featured['location_avg_price'].min()
        max_lap = df_featured['location_avg_price'].max()
        logger.info(f"✅ Created location_avg_price feature")
        logger.info(f"   Range: {min_lap:.2f} - {max_lap:.2f}")
        logger.info(f"   Localities: {len(location_price)}")
    
    return df_featured


def preprocess_data(input_file, output_file):
    """
    Complete data preprocessing pipeline (2 stages).
    
    Stage 1 (Basic):
    1. Load raw data
    2. Drop unnecessary columns
    3. Check for missing values
    4. Validate dataset
    
    Stage 2 (Advanced):
    5. Filter bedroom outliers
    6. Clean locality names
    7. Consolidate rare localities
    8. Create price_per_sqft feature
    9. Create location_avg_price feature
    10. Save final processed data
    
    Args:
        input_file: Path to raw CSV file
        output_file: Path to save final processed CSV file
        
    Returns:
        Final processed DataFrame
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        # ===== STAGE 1: BASIC PREPROCESSING =====
        df = load_data(input_file)
        df = drop_unnecessary_columns(df)
        df = check_missing_values(df)
        df = validate_data(df)
        
        logger.info("")  # Blank line for readability
        
        # ===== STAGE 2: ADVANCED PREPROCESSING =====
        df = filter_bedroom_outliers(df)
        df = clean_locality_names(df)
        df = consolidate_rare_localities(df, threshold=10)
        df = create_price_per_sqft(df)
        df = create_location_avg_price(df)
        
        # Save final processed data
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved final processed data to {output_file}")
        
        # Summary statistics
        logger.info("")  # Blank line for readability
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE - SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Final columns: {list(df.columns)}")
        logger.info(f"Final rows: {df.shape[0]}")
        logger.info(f"Final columns count: {df.shape[1]}")
        logger.info("=" * 60)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error in preprocessing pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess raw housing data for Delhi rent prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python src/data/preprocessing.py \\
        --input data/raw/data.csv \\
        --output data/processed/final_data.csv

This script performs 2-stage preprocessing:
- Stage 1 (Basic): Column removal, validation
- Stage 2 (Advanced): Outlier filtering, feature engineering
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to raw CSV file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save final processed CSV file'
    )
    
    args = parser.parse_args()
    
    # Run the preprocessing pipeline
    preprocess_data(args.input, args.output)