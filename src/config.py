# src/config.py

"""
Project configuration for Pay Gap ML:
- paths to data files
- folders for saved models
- other global constants
"""

from pathlib import Path
import sys

# ---------------------------
# Base directory
# ---------------------------
try:
    # works in regular .py scripts
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # fallback for notebooks
    BASE_DIR = Path.cwd()

# ---------------------------
# Data paths
# ---------------------------
DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_CSV = DATA_DIR / "salary_data_train.csv"
SYNTHETIC_CSV = DATA_DIR / "salary_data_synthetic.csv"

# ---------------------------
# Models folder
# ---------------------------
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists

MODEL_FILES = {
    "linear_regression": MODELS_DIR / "linear_regression.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
    "bayesian_ridge": MODELS_DIR / "bayesian_ridge.pkl",
    "bart": MODELS_DIR / "bart.pkl",
    "tabpfn": MODELS_DIR / "tabpfn.pkl",
    "autogluon": MODELS_DIR / "autogluon.pkl"
}

# ---------------------------
# Other constants
# ---------------------------
RANDOM_STATE = 42
MIN_INCOME = 1276
