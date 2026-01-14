from pathlib import Path

# Base directory of the project (folder 'pay-gap-ml')
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_CSV = DATA_DIR / "salary_data_train.csv"
SYNTHETIC_CSV = DATA_DIR / "salary_data_synthetic.csv"

# Models directory
MODELS_DIR = BASE_DIR / "models"

# Model files
MODEL_FILES = {
    "linear_regression": MODELS_DIR / "linear_regression.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
    "bayesian_ridge": MODELS_DIR / "bayesian_ridge.pkl",
    "bart": MODELS_DIR / "bart.pkl",
    "tabpfn": MODELS_DIR / "tabpfn.pkl",
    "autogluon": MODELS_DIR / "autogluon.pkl"
}

# Global constants
RANDOM_STATE = 42
MIN_INCOME = 1276
