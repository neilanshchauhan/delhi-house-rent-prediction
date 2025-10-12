"""
Final Model Training and Registration Script
============================================
This script trains and optionally registers the final model in MLflow.
MLflow tracking is optional and can be disabled for CI/CD pipelines.

Example Usage:
    # Without MLflow (for CI/CD)
    python train_model.py \
        --config configs/model_config.yaml \
        --data data/processed/featured_house_data.csv \
        --models-dir models/

    # With MLflow (for local development)
    python train_model.py \
        --config configs/model_config.yaml \
        --data data/processed/featured_house_data.csv \
        --models-dir models/ \
        --mlflow-tracking-uri http://localhost:5555
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import yaml
import logging
import platform
import os

# --- Scikit-learn and Model Imports ---
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# --- MLflow Imports (optional) ---
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and register the final model from a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config YAML file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the processed CSV dataset for training"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory to save the final trained model artifact"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (optional, for local development only)"
    )
    return parser.parse_args()

# -----------------------------
# Model Loading from Config
# -----------------------------
def get_model_instance(name, params):
    """Initializes a model instance from its name and parameters."""
    model_map = {
        'Linear Regression': LinearRegression,
        'Lasso': Lasso,
        'Ridge': Ridge,
        'KNN': KNeighborsRegressor,
        'Decision Tree': DecisionTreeRegressor,
        'Random Forest': RandomForestRegressor,
        'Gradient Boosting': GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor,
    }
    
    if name not in model_map:
        raise ValueError(f"Unsupported model specified in config: '{name}'")
    
    # Clean parameters: only keep valid hyperparameters
    valid_params = {
        k: v for k, v in params.items() 
        if k in model_map[name]().get_params()
    }
    
    return model_map[name](**valid_params)

# -----------------------------
# MLflow Helper Functions
# -----------------------------
def setup_mlflow(mlflow_uri, model_name):
    """Setup MLflow if tracking URI is provided."""
    if not mlflow_uri:
        logger.info("MLflow tracking disabled (no tracking URI provided)")
        return False
    
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed, skipping tracking")
        return False
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(model_name)
        logger.info(f"MLflow tracking enabled at {mlflow_uri}")
        return True
    except Exception as e:
        logger.warning(f"Failed to setup MLflow: {str(e)}")
        logger.warning("Continuing without MLflow tracking")
        return False


def log_to_mlflow(model, model_name, model_cfg, mae, r2, run_id):
    """Log model and metrics to MLflow."""
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        logger.info("Logging to MLflow...")
        
        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({'mae': mae, 'r2_score': r2})
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run_id}/model"
        
        # Register model in Model Registry
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
            logger.info(f"Created registered model '{model_name}'")
        except mlflow.exceptions.MlflowException:
            logger.info(f"Model '{model_name}' already exists in registry")
        
        # Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Created model version {model_version.version}")
        
        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"Transitioned model to Staging stage")
        
        # Add description and tags
        description = (
            f"Final trained model for predicting house prices in Delhi. "
            f"Version {model_version.version} trained with {model_cfg['best_model_name']}."
        )
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        
        # Log dependencies
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__
        }
        mlflow.set_tags(deps)
        logger.info(f"Logged dependencies: {deps}")
        
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        logger.warning("Model still saved locally, but MLflow logging failed")

# -----------------------------
# Main Training Logic
# -----------------------------
def main(args):
    """Main function to run the training pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Load config file
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model_details']
    
    # Setup MLflow (optional)
    mlflow_enabled = setup_mlflow(args.mlflow_tracking_uri, model_cfg['name'])
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    target = model_cfg['target_variable']
    features = model_cfg['features_used']
    
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Features: {features}")
    logger.info(f"Target: {target}")
    
    # Prepare features and target
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # Get model instance
    logger.info(f"Loading model: {model_cfg['best_model_name']}")
    model = get_model_instance(model_cfg['best_model_name'], model_cfg['model_parameters'])
    
    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Training complete")
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  R²: {r2:.4f}")
    
    # Log to MLflow (if enabled)
    if mlflow_enabled and MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="final_model_training") as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run ID: {run_id}")
                log_to_mlflow(
                    model,
                    model_cfg['name'],
                    model_cfg,
                    mae,
                    r2,
                    run_id
                )
        except Exception as e:
            logger.error(f"MLflow error: {str(e)}")
            logger.warning("Continuing with local model save only")
    else:
        logger.info("Skipping MLflow logging (not enabled or not available)")
    
    # Save model locally
    save_dir = os.path.join(args.models_dir, "trained")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_cfg['name']}.pkl")
    joblib.dump(model, save_path)
    logger.info(f"Model saved to: {save_path}")
    
    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    args = parse_args()
    main(args)