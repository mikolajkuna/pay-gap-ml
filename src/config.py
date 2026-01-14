# src/mikolajkuna/config.py

"""
Project configuration for Pay Gap ML:
- paths to data files
- folders for saved models
- other global constants
"""

from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_CSV = DATA_DIR / "salary_data_train.csv"
SYNTHETIC_CSV = DATA_DIR / "salary_data_synthetic.csv"

# Directory to store trained models
MODELS_DIR = BASE_DIR / "models"

# Paths to specific model files
MODEL_FILES = {
    "linear_regression": MODELS_DIR / "linear_regression.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
    "bayesian_ridge": MODELS_DIR / "bayesian_ridge.pkl",
    "bart": MODELS_DIR / "bart.pkl",
    "tabpfn": MODELS_DIR / "tabpfn.pkl",
    "autogluon": MODELS_DIR / "autogluon.pkl"
}

# Other global constants
RANDOM_STATE = 42
MIN_INCOME = 1276
