"""
Final Model Training and Registration Script
============================================
This script takes a configuration file and a processed dataset to train,
evaluate, and register the final best model in the MLflow Model Registry.

Example Usage:
    python train_model.py \
        --config configs/best_model_config.yaml \
        --data data/processed/featured_house_data.csv \
        --models-dir models/ \
        --mlflow-tracking-uri sqlite:///mlflow.db
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

# --- MLflow Imports ---
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and register the final model from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the best_model_config.yaml file.")
    parser.add_argument("--data", type=str, required=True, help="Path to the processed CSV dataset for training.")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save the final trained model artifact.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI.")
    return parser.parse_args()

# -----------------------------
# Model Loading from Config
# -----------------------------
def get_model_instance(name, params):
    """Initializes a model instance from its name and parameters."""
    # Expanded map to include all 9 models from our experiments
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
    
    # Clean parameters: remove any non-parameter keys if they exist
    # For example, our YAML includes the model object itself which isn't a valid hyperparameter
    valid_params = {k: v for k, v in params.items() if k in model_map[name]().get_params()}
    
    return model_map[name](**valid_params)

# -----------------------------
# Main Training Logic
# -----------------------------
def main(args):
    """Main function to run the training pipeline."""
    # Load config file
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Adapted to the YAML structure from our previous script
    model_cfg = config['model_details']

    # Set up MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg['name'])
        logger.info(f"MLflow tracking enabled. URI: {args.mlflow_tracking_uri}")

    # Load and prepare data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    target = model_cfg['target_variable']
    features = model_cfg['features_used']

    # Robust feature selection based on the config file
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get model instance
    model = get_model_instance(model_cfg['best_model_name'], model_cfg['model_parameters'])

    # Start MLflow run
    with mlflow.start_run(run_name="final_model_training") as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow run: {run_id}")
        logger.info(f"Training final model: {model_cfg['best_model_name']}")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        logger.info(f"Evaluation metrics - MAE: {mae:.2f}, R²: {r2:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({'mae': mae, 'r2_score': r2})

        # Log and register model
        model_name = model_cfg['name']
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run_id}/model"
        
        logger.info(f"Registering model '{model_name}' to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            logger.warning(f"Registered model '{model_name}' already exists.")

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Created model version {model_version.version} for model '{model_name}'")

        # Transition model to "Staging"
        logger.info("Transitioning model version to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Add description and tags
        description = (
            f"Final trained model for predicting house prices in Delhi. "
            f"This is version {model_version.version} of the '{model_name}' model, "
            f"trained with the {model_cfg['best_model_name']} algorithm."
        )
        client.update_model_version(name=model_name, version=model_version.version, description=description)

        # Log library versions as tags for reproducibility
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__
        }
        mlflow.set_tags(deps)
        logger.info(f"Logged dependencies as tags: {deps}")

        # Save model locally
        save_dir = os.path.join(args.models_dir, "trained")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(model, save_path)
        logger.info(f"✅ Final model saved locally to: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)